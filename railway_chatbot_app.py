import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------- FAQs ---------------
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

# --------------- Streamlit UI ---------------
st.set_page_config(page_title="Indian Railways Chatbot", page_icon="🚆")

st.markdown("""
    <style>
        .chat-bubble {
            border-radius: 12px;
            padding: 12px 18px;
            margin: 8px 0;
            max-width: 80%;
        }
        .user {
            background-color: #cce5ff;
            color: #003366;
            margin-left: auto;
            text-align: right;
        }
        .bot {
            background-color: #e2e3e5;
            color: #333;
            margin-right: auto;
            text-align: left;
        }
        .title {
            text-align: center;
            color: #2c3e50;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2 class='title'>🚆 Indian Railways Chatbot</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Ask your questions about tickets, PNR, delays, and more</p>", unsafe_allow_html=True)

# Initialize history
if "history" not in st.session_state:
    st.session_state.history = []

# Input form
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask a railway-related question", placeholder="e.g. enquiry through sms")
    submitted = st.form_submit_button("Ask")

def get_response(query):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(questions + [query])
    similarity = cosine_similarity(vectors[-1], vectors[:-1])
    idx = similarity.argmax()
    score = similarity[0, idx]
    if score > 0.3:  # good threshold
        return answers[idx]
    return "😕 Sorry, I couldn’t understand that. Try rephrasing your question."

# Handle submission
if submitted and user_input:
    response = get_response(user_input)
    st.session_state.history.append(("user", user_input))
    st.session_state.history.append(("bot", response))

# Show chat
for sender, msg in st.session_state.history:
    css_class = "user" if sender == "user" else "bot"
    st.markdown(f"<div class='chat-bubble {css_class}'><b>{'You' if sender=='user' else 'Bot'}:</b> {msg}</div>", unsafe_allow_html=True)
