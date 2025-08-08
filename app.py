import os
import subprocess
import streamlit as st
import tempfile
import yt_dlp
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import shutil

# Directories
os.makedirs("downloads", exist_ok=True)
os.makedirs("temp_audio", exist_ok=True)

# HuggingFace Model for QA
model_name = "MBZUAI/LaMini-Flan-T5-783M"
qa_pipeline = pipeline("text2text-generation", model=model_name)
llm = HuggingFacePipeline(pipeline=qa_pipeline)

# Embeddings
embeddings = HuggingFaceEmbeddings()

# Store transcript & index in session state
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

st.title("🎬 YouTube Insight AI")

st.markdown("""
**Note on YouTube URLs**:  
YouTube often blocks direct downloads (Error 403) due to restrictions.  
Uploads work 100% of the time, so for testing please upload your own file.  
""")

# ========== DOWNLOAD FROM YOUTUBE ==========
def download_from_youtube(url, audio_only=False):
    try:
        ydl_opts = {
            "outtmpl": "downloads/%(title)s.%(ext)s",
            "format": "bestaudio/best" if audio_only else "best",
            "quiet": True,
            "noplaylist": True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info)
    except Exception as e:
        st.error(f"[ERROR] {str(e)}")
        return None

# ========== AUDIO EXTRACTION ==========
def extract_audio(video_path):
    try:
        audio_path = os.path.join("temp_audio", "audio.mp3")
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "libmp3lame", audio_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return audio_path
    except Exception as e:
        st.error(f"[ERROR] Audio extraction failed: {e}")
        return None

# ========== TRANSCRIBE ==========
def transcribe_audio(audio_path):
    try:
        transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-small")
        result = transcriber(audio_path)
        return result["text"]
    except Exception as e:
        st.error(f"[ERROR] {e}")
        return None

# ========== BUILD VECTOR STORE ==========
def build_vectorstore(transcript):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(transcript)
    docs = [Document(page_content=chunk) for chunk in chunks]
    return FAISS.from_documents(docs, embeddings)

# ========== INPUT OPTIONS ==========
url_input = st.text_input("Paste YouTube URL here:")
uploaded_video = st.file_uploader("Or upload a video file", type=["mp4", "mov", "avi", "mkv"])

col1, col2 = st.columns(2)
with col1:
    if st.button("Download Audio from YouTube"):
        if url_input:
            video_path = download_from_youtube(url_input, audio_only=True)
            if video_path:
                st.success("Audio downloaded successfully.")
        else:
            st.warning("Please enter a YouTube URL.")

with col2:
    if st.button("Upload & Extract Audio"):
        if uploaded_video:
            temp_path = os.path.join("downloads", uploaded_video.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_video.read())
            audio_path = extract_audio(temp_path)
            if audio_path:
                st.session_state.audio_path = audio_path
                st.success("Audio extracted successfully.")
                with open(audio_path, "rb") as f:
                    st.download_button("Download Extracted Audio", f, "audio.mp3")
        else:
            st.warning("Please upload a video file.")

# ========== TRANSCRIBE BUTTON ==========
if st.button("Transcribe Audio"):
    audio_path = None
    if uploaded_video and "audio_path" in st.session_state:
        audio_path = st.session_state.audio_path
    elif url_input:
        audio_path = download_from_youtube(url_input, audio_only=True)

    if audio_path and os.path.exists(audio_path):
        transcript = transcribe_audio(audio_path)
        if transcript:
            st.session_state.transcript = transcript
            st.session_state.vectorstore = build_vectorstore(transcript)
            st.success("Transcription complete!")
    else:
        st.error("[ERROR] Audio file not found. Please download or upload first.")

# ========== DISPLAY TRANSCRIPT ==========
if st.session_state.transcript:
    st.subheader("Transcript")
    st.text_area("Transcript", value=st.session_state.transcript, height=300)

# ========== QA BOX ==========
st.subheader("Ask a Question about the Transcript")
st.caption("Note: Please transcribe audio first before asking questions.")
question = st.text_input("Your Question:")
if st.button("Get Answer"):
    if st.session_state.vectorstore:
        retriever = st.session_state.vectorstore.as_retriever()
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        answer = qa.run(question)
        st.write("**Answer:**", answer)
    else:
        st.warning("Please transcribe audio first before asking questions.")
