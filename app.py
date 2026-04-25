import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from langdetect import detect
from deep_translator import GoogleTranslator
import numpy as np
from gtts import gTTS
from io import BytesIO

# --- Page Config ---
st.set_page_config(page_title="Rural Health AI", page_icon="🏥")

# --- Model Loading (Cached for Performance) ---
@st.cache_resource
def load_models():
    # Embedding model
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # QA Pipeline
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    return embedder, qa_pipeline

embedder, qa_pipeline = load_models()

# --- Knowledge Base ---
KB_PARAS = [
    "Paracetamol reduces fever and relieves mild pain. Typical adult dose is 500–1000 mg every 4–6 hours, max 4 g/day.",
    "Insulin must be stored between 2°C and 8°C. Monitor blood glucose regularly.",
    "A balanced rural diet includes rice, pulses (dal), vegetables, and safe drinking water.",
    "Chest pain with sweating and breathlessness is an emergency. Go to a hospital immediately.",
    "Dehydration in children: signs include sunken eyes and dry mouth. Give ORS immediately.",
    "Malaria symptoms: fever, chills, and headache. Requires prompt medical testing.",
    "Wash hands with soap before food and after toilet use to prevent diseases."
]

@st.cache_data
def get_kb_embeddings():
    return embedder.encode(KB_PARAS, convert_to_numpy=True)

kb_emb = get_kb_embeddings()

# --- Helper Functions ---
def translate_text(text, src="auto", dst="en"):
    try:
        return GoogleTranslator(source=src, target=dst).translate(text)
    except:
        return text

def get_answer(question_en):
    # Context Retrieval
    q_emb = embedder.encode([question_en], convert_to_numpy=True)[0]
    cos_sim = np.dot(kb_emb, q_emb) / (np.linalg.norm(kb_emb, axis=1) * np.linalg.norm(q_emb))
    ctx = KB_PARAS[np.argmax(cos_sim)]
    
    # QA logic
    result = qa_pipeline({"question": question_en, "context": ctx})
    return result['answer'], result['score'], ctx

# --- Streamlit UI ---
st.title("🏥 Rural Health Information Assistant")
st.markdown("Ask medical questions in **English** or **Telugu**.")

user_input = st.text_input("Enter your question:", placeholder="e.g., What to do for fever?")

if user_input:
    # 1. Detect Language
    try:
        lang = detect(user_input)
    except:
        lang = "en"
    
    # 2. Translate if Telugu
    query_en = translate_text(user_input, src=lang, dst="en") if lang == "te" else user_input
    
    # 3. Get Answer
    with st.spinner("Analyzing..."):
        ans, score, context = get_answer(query_en)
        
        if score < 0.1:
            final_ans = f"I'm not sure, but here is some related info: {context}"
        else:
            final_ans = ans
            
        # Add safety warnings
        final_ans += "\n\n**Note:** Informational only. Consult a doctor for medical decisions."

        # 4. Translate back if needed
        output_text = translate_text(final_ans, src="en", dst="te") if lang == "te" else final_ans
        
    # 5. Display Results
    st.subheader("Response:")
    st.write(output_text)
    
    # 6. Audio Playback
    tts = gTTS(text=output_text, lang=lang)
    audio_fp = BytesIO()
    tts.write_to_fp(audio_fp)
    st.audio(audio_fp, format="audio/mp3")

st.info("Emergency: If you have chest pain or breathing trouble, go to the nearest hospital immediately.")
