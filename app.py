import os
import yt_dlp
import whisper
import zipfile
import streamlit as st

#from moviepy.editor import VideoFileClip
from urllib.parse import urlparse, parse_qs

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

# ----------- Utils -----------
os.makedirs("downloads", exist_ok=True)
qa_bot = None

def clean_youtube_url(url):
    try:
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        if 'v' in query_params:
            video_id = query_params['v'][0]
            return f"https://www.youtube.com/watch?v={video_id}"
        return url
    except:
        return url

def download_media(url, media_type):
    try:
        url = clean_youtube_url(url)
        output_path = "downloads/video.mp4" if media_type == "video" else "downloads/audio.mp3"

        ydl_opts = {
            "format": "best[ext=mp4]/best" if media_type == "video" else "bestaudio[ext=m4a]/bestaudio",
            "outtmpl": output_path,
            "quiet": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Zip the downloaded file
        zip_path = output_path + ".zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(output_path, arcname=os.path.basename(output_path))

        return zip_path, "✅ File retrieved successfully (zipped)."
    except Exception as e:
        return None, f"[ERROR] {str(e)}"


import subprocess

def extract_audio(video_path, output_audio_path="downloads/audio.mp3"):
    try:
        command = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn",  # no video
            "-acodec", "libmp3lame",
            "-ab", "192k",
            output_audio_path
        ]
        subprocess.run(command, check=True)
        if os.path.exists(output_audio_path):
            return output_audio_path
        else:
            raise FileNotFoundError("FFmpeg did not produce the audio file.")
    except Exception as e:
        print(f"[extract_audio ERROR] {e}")
        return None



def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result['text']

def create_faiss_index(transcript_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    docs = text_splitter.create_documents([transcript_text])
    embeddings = HuggingFaceEmbeddings()
    index = FAISS.from_documents(docs, embeddings)
    return index

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain

def build_qa_bot(index):
    retriever = index.as_retriever(top_k=6)

    prompt_template = """You are an expert video assistant helping users understand content from YouTube videos.

Here is a portion of the transcript:
{context}

Using only the information above, clearly and concisely answer the following question in your own words.

If the answer is not directly available, respond with:
"The video does not mention that."

Question:
{question}

Answer:"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    # Get the OpenRouter key from Streamlit secrets or env
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")  # Store this in Streamlit Secrets

    llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://youtubeinsightai.streamlit.app/",  # <- Replace with your actual Streamlit app URL
            "X-Title": "YouTube Insight AI",
        },
        temperature=0.7,
        max_tokens=1024
    )

    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    qa = RetrievalQA(combine_documents_chain=chain, retriever=retriever, return_source_documents=False)
    return qa



# ----------- Streamlit UI -----------

st.set_page_config(page_title="YouTube Insight AI", layout="centered")
st.title("🎥 YouTube Insight AI")

url = st.text_input("Enter YouTube Video URL")

media_type = st.radio("Choose download type", ["video", "audio"])
if st.button("Download"):
    with st.spinner("Downloading..."):
        file_path, status = download_media(url, media_type)
    
    if file_path is not None:
        st.success(status)
        with open(file_path, "rb") as f:
            st.download_button("Download ZIP", f, file_name=os.path.basename(file_path))
    else:
        st.error(status)

if st.button("Transcribe Audio"):
    with st.spinner("Transcribing..."):
        try:
            audio_path = "downloads/audio.mp3"
            if not os.path.exists(audio_path):
                st.error("[ERROR] Audio file not found. Please download as 'audio' first.")
            else:
                transcript = transcribe_audio(audio_path)
                db = create_faiss_index(transcript)
                st.session_state["qa_bot"] = build_qa_bot(db)
                st.session_state["transcript"] = transcript
                st.success("Transcription completed!")
                st.text_area("Transcript", transcript, height=250)
        except Exception as e:
            st.error(f"[ERROR] {str(e)}")
st.warning("To transcribe, please download the video as 'audio' first.")



if "transcript" in st.session_state:
    question = st.text_input("Ask a Question")
    if st.button("Get Answer"):
        if "qa_bot" in st.session_state:
            with st.spinner("Thinking..."):
                answer = st.session_state["qa_bot"].run(question)
            st.text_area("Answer", answer, height=200)
        else:
            st.warning("Please transcribe a video first.")

