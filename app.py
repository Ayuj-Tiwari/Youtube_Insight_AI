# app.py
import os
import subprocess
import traceback
import zipfile
from urllib.parse import urlparse, parse_qs

import streamlit as st
import yt_dlp
import whisper

# langchain-community imports (use community variants)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

from transformers import pipeline

# ----------------- Basic setup -----------------
st.set_page_config(page_title="YouTube Insight AI", layout="centered")
st.title("🎥 YouTube Insight AI ")

os.makedirs("downloads", exist_ok=True)
os.makedirs("temp_audio", exist_ok=True)

# If the user stored cookies in Streamlit secrets (TOML),
# write cookies.txt locally for yt-dlp to use.
if "cookies" in st.secrets and st.secrets["cookies"].strip():
    try:
        with open("cookies.txt", "w", encoding="utf-8") as f:
            f.write(st.secrets["cookies"])
    except Exception as e:
        st.warning("Failed to write cookies from secrets (will continue without cookies).")

# ----------------- Session defaults -----------------
if "transcript" not in st.session_state:
    st.session_state["transcript"] = None
if "qa_bot" not in st.session_state:
    st.session_state["qa_bot"] = None
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "audio_path" not in st.session_state:
    st.session_state["audio_path"] = None
if "video_path" not in st.session_state:
    st.session_state["video_path"] = None
if "uploaded_audio_path" not in st.session_state:
    st.session_state["uploaded_audio_path"] = None

# ----------------- Helpers -----------------
def clean_youtube_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return f"https://www.youtube.com/watch?v={qs['v'][0]}"
    except Exception:
        pass
    return url

def run_ffmpeg_to_mp3(src: str, dst: str) -> str | None:
    """Convert src to MP3 at dst using ffmpeg. Returns dst on success else None."""
    try:
        cmd = ["ffmpeg", "-y", "-i", src, "-vn", "-acodec", "libmp3lame", "-ab", "192k", dst]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return dst if os.path.exists(dst) else None
    except Exception as e:
        print("[ffmpeg error]", e)
        return None

def download_from_youtube(url: str, media_type: str = "video") -> str | None:
    """
    Downloads from YouTube using yt-dlp.
    Returns a file path on success or None on failure.
    """
    try:
        url = clean_youtube_url(url)
        # flexible format selection (avoid exact ext requirements)
        if media_type == "audio":
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": "downloads/%(title)s.%(ext)s",
                "quiet": True,
                "noplaylist": True,
                "nocheckcertificate": True,
            }
        else:
            ydl_opts = {
                "format": "bestvideo+bestaudio/best",
                "outtmpl": "downloads/%(title)s.%(ext)s",
                "quiet": True,
                "noplaylist": True,
                "merge_output_format": "mp4",
                "nocheckcertificate": True,
            }

        # Use cookies if present
        if os.path.exists("cookies.txt"):
            ydl_opts["cookiefile"] = "cookies.txt"

        # Add headers to reduce chance of 403
        ydl_opts["headers"] = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.youtube.com"
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filepath = ydl.prepare_filename(info)

        # If audio requested, convert to MP3 for Whisper compatibility
        if media_type == "audio":
            if filepath.lower().endswith(".mp3"):
                return filepath
            mp3_path = os.path.join("downloads", "audio.mp3")
            converted = run_ffmpeg_to_mp3(filepath, mp3_path)
            if converted:
                return converted
            # fallback: return original if conversion failed
            return filepath if os.path.exists(filepath) else None
        else:
            return filepath if os.path.exists(filepath) else None

    except Exception as e:
        traceback.print_exc()
        st.error(f"[Download error] {str(e)}")
        return None

# Cached Whisper model loader
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

def transcribe_audio(audio_path: str) -> str | None:
    try:
        model = load_whisper_model()
        # whisper returns a dict with "text"
        res = model.transcribe(audio_path)
        return res.get("text", "")
    except Exception as e:
        st.error(f"[Transcription error] {e}")
        return None

def build_faiss_from_transcript(transcript: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    docs = splitter.create_documents([transcript])
    embed = HuggingFaceEmbeddings()
    idx = FAISS.from_documents(docs, embed)
    return idx

def build_qa_bot(index):
    retriever = index.as_retriever(top_k=6)
    prompt_template = """You are an expert video assistant. Answer using only the transcript text given.

Transcript snippet:
{context}

If the answer is not present, reply: "The video does not mention that."

Question:
{question}

Answer:"""
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    # HuggingFace pipeline model (light-ish)
    hf_pipe = pipeline(
        "text2text-generation",
        model="MBZUAI/LaMini-Flan-T5-783M",
        tokenizer="MBZUAI/LaMini-Flan-T5-783M",
        max_length=512,
        repetition_penalty=1.2,
        temperature=0.7
    )
    llm = HuggingFacePipeline(pipeline=hf_pipe)
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    qa = RetrievalQA(combine_documents_chain=chain, retriever=retriever, return_source_documents=False)
    return qa

# ----------------- UI -----------------
st.markdown(
    "⚠️ **YouTube note:** Some YouTube links may fail (403 Forbidden) due to Youtube's strict access restrictions. "
    "If you encounter download errors, upload a video file (MP4/MKV/MOV/AVI) instead — uploads always work. "
)

# --- Inputs: YouTube URL + buttons
url = st.text_input("Paste YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
c1, c2 = st.columns(2)

with c1:
    if st.button("Download Video from YouTube"):
        if not url:
            st.warning("Paste a YouTube URL first.")
        else:
            st.info("Downloading video — this may fail for some links (see note above).")
            vfile = download_from_youtube(url, media_type="video")
            if vfile:
                st.session_state["video_path"] = vfile
                st.success(f"Downloaded video: {os.path.basename(vfile)}")
                with open(vfile, "rb") as fh:
                    st.download_button("⬇️ Download Video (local copy)", fh, file_name=os.path.basename(vfile))
            else:
                st.error("Video download failed. Consider uploading the file or adding cookies to secrets for a demo.")

with c2:
    if st.button("Download Audio from YouTube"):
        if not url:
            st.warning("Paste a YouTube URL first.")
        else:
            st.info("Downloading audio — this may fail for some links (see note above).")
            afile = download_from_youtube(url, media_type="audio")
            if afile:
                st.session_state["audio_path"] = afile
                st.success(f"Downloaded audio: {os.path.basename(afile)}")
                with open(afile, "rb") as fh:
                    st.download_button("⬇️ Download Audio (local copy)", fh, file_name=os.path.basename(afile))
            else:
                st.error("Audio download failed. Consider uploading the file or adding cookies to secrets for a demo.")

st.markdown("---")
st.subheader("Or upload a video file (recommended for recruiters/testers)")
uploaded = st.file_uploader("Upload video (MP4/MKV/MOV/AVI)", type=["mp4", "mkv", "mov", "avi"])
if uploaded:
    dest = os.path.join("downloads", uploaded.name)
    with open(dest, "wb") as out:
        out.write(uploaded.read())
    st.success(f"Saved uploaded file: {uploaded.name}")
    # extract audio to downloads/audio.mp3 (preferred)
    converted = run_ffmpeg_to_mp3(dest, os.path.join("downloads", "audio.mp3"))
    if converted:
        st.session_state["uploaded_audio_path"] = converted
        st.session_state["audio_path"] = converted
        st.success("Extracted audio from uploaded video. You can download it below.")
        with open(converted, "rb") as fh:
            st.download_button("⬇️ Download Extracted Audio", fh, file_name="extracted_audio.mp3")
    else:
        st.error("Audio extraction failed (ffmpeg may not be available).")

st.markdown("---")
st.info("To transcribe, ensure you have an audio file (downloaded or uploaded). Then click Transcribe.")

# Transcribe button
if st.button("Transcribe Audio"):
    audio_candidate = st.session_state.get("audio_path") or st.session_state.get("uploaded_audio_path")
    # if only video present, try to extract
    if not audio_candidate and st.session_state.get("video_path"):
        try:
            audio_candidate = run_ffmpeg_to_mp3(st.session_state["video_path"], os.path.join("downloads", "audio.mp3"))
            if audio_candidate:
                st.session_state["audio_path"] = audio_candidate
        except Exception as e:
            print("[ffmpeg extract error]", e)

    if not audio_candidate or not os.path.exists(audio_candidate):
        st.error("No audio found — download audio or upload a video first.")
    else:
        st.info("Transcribing (Whisper base) — this may take some time.")
        transcript_text = transcribe_audio(audio_candidate)
        if transcript_text is None:
            st.error("Transcription failed.")
        else:
            st.session_state["transcript"] = transcript_text
            st.success("Transcription complete — building index (FAISS).")
            try:
                idx = build_faiss_from_transcript(transcript_text)
                st.session_state["vectorstore"] = idx
                st.session_state["qa_bot"] = build_qa_bot(idx)
                st.success("Index and QA bot ready.")
            except Exception as e:
                st.error(f"Index/build failed: {e}")

# Always display transcript if present
if st.session_state.get("transcript"):
    st.subheader("📜 Transcript")
    st.text_area("Transcript", value=st.session_state["transcript"], height=300)

st.markdown("---")
st.subheader("❓ Ask a Question about the transcript")
st.caption("Q&A is always visible but will only return answers after you transcribe the audio.")
question = st.text_input("Enter a question")

if st.button("Get Answer"):
    if st.session_state.get("qa_bot") and st.session_state.get("vectorstore"):
        with st.spinner("Thinking..."):
            try:
                ans = st.session_state["qa_bot"].run(question)
                st.markdown("**Answer:**")
                st.write(ans)
            except Exception as e:
                st.error(f"QA error: {e}")
    else:
        st.warning("Please transcribe audio first to enable Q&A.")
