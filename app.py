# app.py
import os
import subprocess
import zipfile
import traceback

import streamlit as st
import yt_dlp
import whisper

from urllib.parse import urlparse, parse_qs

# langchain-community imports (consistent)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

from transformers import pipeline

# ----------------- Setup -----------------
st.set_page_config(page_title="YouTube Insight AI", layout="centered")
st.title("🎥 YouTube Insight AI")

os.makedirs("downloads", exist_ok=True)
os.makedirs("temp_audio", exist_ok=True)

# Optional: write cookies if provided in Streamlit secrets (TOML: cookies = """...""" )
if "cookies" in st.secrets:
    try:
        with open("cookies.txt", "w") as f:
            f.write(st.secrets["cookies"])
    except Exception as e:
        st.warning("Could not write cookies.txt from secrets. Continuing without cookies.")

# Session-state defaults
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
            vid = qs["v"][0]
            return f"https://www.youtube.com/watch?v={vid}"
    except Exception:
        pass
    return url

def run_ffmpeg_convert_to_mp3(src_path: str, dst_path: str) -> str | None:
    """Convert any audio/video file to MP3 using ffmpeg. Returns dst_path or None."""
    try:
        cmd = [
            "ffmpeg", "-y", "-i", src_path,
            "-vn", "-acodec", "libmp3lame", "-ab", "192k", dst_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return dst_path if os.path.exists(dst_path) else None
    except Exception as e:
        print("[ffmpeg error]", e)
        return None

def download_from_youtube(url: str, media_type: str = "video") -> str | None:
    """
    Downloads a youtube video or audio using yt_dlp.
    Returns the downloaded file path or None on failure.
    """
    try:
        url = clean_youtube_url(url)
        # Use flexible formats so we don't require a specific extension
        if media_type == "audio":
            outtmpl = "downloads/%(title)s.%(ext)s"
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": outtmpl,
                "quiet": True,
                "noplaylist": True,
                "nocheckcertificate": True,
                "source_address": "0.0.0.0"
            }
        else:
            outtmpl = "downloads/%(title)s.%(ext)s"
            ydl_opts = {
                "format": "bestvideo+bestaudio/best",
                "outtmpl": outtmpl,
                "quiet": True,
                "noplaylist": True,
                "merge_output_format": "mp4",
                "nocheckcertificate": True,
                "source_address": "0.0.0.0"
            }

        # If cookies.txt exists (written from secrets), pass it
        if os.path.exists("cookies.txt"):
            ydl_opts["cookiefile"] = "cookies.txt"

        # Add browser-like headers to help reduce 403's
        ydl_opts["headers"] = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.youtube.com"
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filepath = ydl.prepare_filename(info)

        # For audio: convert to MP3 (whisper expects mp3 works fine)
        if media_type == "audio":
            # if downloaded already mp3, just return; else convert
            if filepath.lower().endswith(".mp3"):
                return filepath
            mp3_path = os.path.join("downloads", "audio.mp3")
            converted = run_ffmpeg_convert_to_mp3(filepath, mp3_path)
            if converted:
                return converted
            else:
                # fallback: return original file if conversion failed
                return filepath if os.path.exists(filepath) else None
        else:
            # video: return final file path
            return filepath if os.path.exists(filepath) else None

    except Exception as e:
        traceback.print_exc()
        st.error(f"[Download error] {e}")
        return None

# Whisper model caching (loads once)
@st.cache_resource
def get_whisper_model():
    return whisper.load_model("base")

def transcribe_with_whisper(audio_path: str) -> str | None:
    try:
        model = get_whisper_model()
        result = model.transcribe(audio_path)
        return result.get("text", "")
    except Exception as e:
        st.error(f"[Transcription error] {e}")
        return None

def create_faiss_index_from_transcript(transcript_text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    docs = splitter.create_documents([transcript_text])
    embedder = HuggingFaceEmbeddings()  # default model; can be specified
    index = FAISS.from_documents(docs, embedder)
    return index

def build_qa_bot_from_index(index):
    retriever = index.as_retriever(top_k=6)

    prompt_template = """You are an expert video assistant helping users understand content from the transcript below.

Transcript excerpt:
{context}

Using **only** the information above, answer the user's question clearly and in your own words.
If the answer is not present, reply: "The video does not mention that."

Question:
{question}

Answer:"""

    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    # Build HuggingFace pipeline (created lazily when QA bot is built)
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
st.markdown("⚠️ **Note:** YouTube downloads can fail (403) because of access restrictions. "
            "For reliable testing, upload your own video file. If you understand the cookie workaround you can add `cookies` to Streamlit Secrets (TOML) — see README.")

# Inputs
url = st.text_input("YouTube URL (optional):")
col1, col2 = st.columns(2)

with col1:
    if st.button("Download Video from YouTube"):
        if not url:
            st.warning("Please paste a YouTube URL first.")
        else:
            st.info("Downloading video (may fail for some links)...")
            vpath = download_from_youtube(url, media_type="video")
            if vpath:
                st.session_state["video_path"] = vpath
                st.success(f"Video downloaded: {os.path.basename(vpath)}")
                # allow download
                with open(vpath, "rb") as f:
                    st.download_button("⬇️ Download Video", f, file_name=os.path.basename(vpath))

with col2:
    if st.button("Download Audio from YouTube"):
        if not url:
            st.warning("Please paste a YouTube URL first.")
        else:
            st.info("Downloading audio (may fail for some links)...")
            apath = download_from_youtube(url, media_type="audio")
            if apath:
                st.session_state["audio_path"] = apath
                st.success(f"Audio downloaded: {os.path.basename(apath)}")
                with open(apath, "rb") as f:
                    st.download_button("⬇️ Download Audio", f, file_name=os.path.basename(apath))

st.markdown("---")
st.subheader("Or upload a video file (MP4, MKV, MOV, etc.)")
uploaded = st.file_uploader("Upload video to extract audio and transcribe", type=["mp4","mkv","mov","avi"])
if uploaded is not None:
    upload_path = os.path.join("downloads", uploaded.name)
    with open(upload_path, "wb") as out:
        out.write(uploaded.read())
    st.success(f"Saved upload: {uploaded.name}")
    # extract audio from uploaded video to downloads/audio.mp3
    extracted = run_ffmpeg_convert_to_mp3(upload_path, os.path.join("downloads","audio.mp3"))
    if extracted:
        st.session_state["uploaded_audio_path"] = extracted
        st.session_state["audio_path"] = extracted
        st.success("Audio extracted and available for download.")
        with open(extracted, "rb") as f:
            st.download_button("⬇️ Download Extracted Audio", f, file_name="extracted_audio.mp3")
    else:
        st.error("Audio extraction failed. ffmpeg may not be available on this host.")

st.markdown("---")

# Transcribe button
if st.button("Transcribe Audio"):
    # choose available audio: prefer session.audio_path -> session.uploaded_audio_path -> try to create from video_path
    audio_candidate = st.session_state.get("audio_path") or st.session_state.get("uploaded_audio_path")
    if not audio_candidate and st.session_state.get("video_path"):
        # try to extract audio from downloaded video
        possible = run_ffmpeg_convert_to_mp3(st.session_state["video_path"], os.path.join("downloads","audio.mp3"))
        if possible:
            audio_candidate = possible
            st.session_state["audio_path"] = possible

    if not audio_candidate or not os.path.exists(audio_candidate):
        st.error("Audio file not found — please download audio from YouTube or upload a video first.")
    else:
        st.info("Transcribing — this can take a while (Whisper model loading).")
        transcript_text = transcribe_with_whisper(audio_candidate)
        if transcript_text:
            st.session_state["transcript"] = transcript_text
            # build index
            try:
                idx = create_faiss_index_from_transcript(transcript_text)
                st.session_state["vectorstore"] = idx
                st.session_state["qa_bot"] = build_qa_bot_from_index(idx)
                st.success("Transcription and index built.")
            except Exception as e:
                st.error(f"Transcribed, but failed to build index: {e}")

# Always show transcript if available
if st.session_state.get("transcript"):
    st.subheader("📜 Transcript")
    st.text_area("Transcript", st.session_state["transcript"], height=300)

st.markdown("---")

# Q&A UI (always visible)
st.subheader("❓ Ask a Question (about the transcript)")
st.caption("Note: the Q&A will work only after transcription is complete.")
question = st.text_input("Enter your question here")
if st.button("Get Answer"):
    if st.session_state.get("qa_bot") and st.session_state.get("vectorstore"):
        with st.spinner("Thinking..."):
            try:
                # use the QA object stored in session
                answer = st.session_state["qa_bot"].run(question)
                st.markdown("**Answer:**")
                st.write(answer)
            except Exception as e:
                st.error(f"QA failed: {e}")
    else:
        st.warning("Please transcribe audio first to enable Q&A.")
