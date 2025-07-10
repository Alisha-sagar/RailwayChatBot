import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from translate import Translator
import pandas as pd
from datetime import datetime

# ---------- CONFIG & STYLE ----------
st.set_page_config(page_title="Indian Railways Chatbot", page_icon="🚆")

st.markdown("""
    <style>
        .chat-bubble {
            border-radius: 16px;
            padding: 12px 20px;
            margin: 10px 0;
            max-width: 80%;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }
        .user {
            background: linear-gradient(to right, #d4fc79, #96e6a1);
            color: #003300;
            margin-left: auto;
            text-align: right;
        }
        .bot {
            background: linear-gradient(to right, #f5f7fa, #c3cfe2);
            color: #333;
            margin-right: auto;
            text-align: left;
        }
        .title {
            text-align: center;
            color: #2c3e50;
            font-size: 32px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2 class='title'>🚆 Indian Railways Chatbot</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>💬 Ask anything about train tickets, delays, PNR & more!</p>", unsafe_allow_html=True)

# ---------- FAQs ----------
faq_data = {
    "How can I book a train ticket online?": "💳 You can book train tickets online through [IRCTC](https://www.irctc.co.in).",
    "Why is PNR not available at night?": "🌙 PNR status and seat availability are unavailable daily from 11:30 PM to 12:30 AM due to system updates.",
    "How to get train enquiry via SMS?": "📱 Send an SMS with your train number to 139 to get train status.",
    "Why is the IRCTC site slow or fonts broken?": "🖥️ Use updated browsers like Chrome/Edge. Some issues occur on older versions.",
    "What happens when train is delayed?": "🚦 You can check [NTES](https://enquiry.indianrail.gov.in/ntes/) for live delay updates.",
    "Where to give railway complaints or suggestions?": "📩 Use the Feedback page on [indianrail.gov.in](https://indianrail.gov.in)."
}
questions = list(faq_data.keys())
answers = list(faq_data.values())

# ---------- SESSION STATE ----------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- TRANSLATION ----------
def translate_to_english(text):
    try:
        translator = Translator(to_lang="en")
        return translator.translate(text)
    except Exception:
        return text  # fallback if translation fails

# ---------- RESPONSE ----------
def get_response(query):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(questions + [query])
    similarity = cosine_similarity(vectors[-1], vectors[:-1])
    idx = similarity.argmax()
    score = similarity[0, idx]
    if score > 0.3:
        return answers[idx]
    return "😕 Sorry, I couldn’t understand that. Try rephrasing your question."

# ---------- LOGGING ----------
def log_query(user_query):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([[now, user_query]], columns=["timestamp", "query"])
    try:
        df.to_csv("query_logs.csv", mode="a", header=not pd.read_csv("query_logs.csv").empty, index=False)
    except:
        df.to_csv("query_logs.csv", mode="a", header=True, index=False)

# ---------- CHAT FORM ----------
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask a railway-related question", placeholder="e.g. train enquiry via SMS")
    submitted = st.form_submit_button("Ask")

if submitted and user_input:
    translated_input = translate_to_english(user_input)
    bot_reply = get_response(translated_input)
    st.session_state.history.append(("user", user_input))
    st.session_state.history.append(("bot", bot_reply))
    log_query(user_input)

# ---------- DISPLAY CHAT ----------
for sender, message in st.session_state.history:
    role = "You" if sender == "user" else "Bot"
    css_class = "user" if sender == "user" else "bot"
    st.markdown(f"<div class='chat-bubble {css_class}'><b>{role}:</b> {message}</div>", unsafe_allow_html=True)

# ---------- FEEDBACK ----------
st.markdown("---")
st.subheader("📮 Feedback")
with st.form("feedback_form"):
    rating = st.slider("Rate this chatbot", 1, 5, 3)
    comment = st.text_input("Any suggestions or comments?")
    submitted_feedback = st.form_submit_button("Submit Feedback")

if submitted_feedback:
    st.success("✅ Thank you for your feedback!")
    feedback_df = pd.DataFrame([[datetime.now().strftime("%Y-%m-%d %H:%M:%S"), rating, comment]],
                               columns=["timestamp", "rating", "comment"])
    try:
        feedback_df.to_csv("feedback.csv", mode="a", header=not pd.read_csv("feedback.csv").empty, index=False)
    except:
        feedback_df.to_csv("feedback.csv", mode="a", header=True, index=False)
