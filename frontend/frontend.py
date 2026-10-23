"""SentimentAI - Streamlit frontend. It only calls the API; it never loads the model."""
import os

import streamlit as st

from api_client import ApiError, predict_sentiment

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "60"))
MAX_CHARS = 5000

EXAMPLES = [
    "This product is excellent and I love it.",
    "The delivery was late and the box was damaged.",
    "What a wonderful movie, I enjoyed every minute!",
    "I am so disappointed, this app keeps crashing.",
]

st.set_page_config(page_title="SentimentAI", page_icon="💬", layout="centered")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #f7f5ff 0%, #eef4ff 100%); }
    .hero { text-align:center; padding: 1.2rem 0 0.4rem 0; }
    .hero h1 { font-size: 2.6rem; margin-bottom: 0.2rem;
               background: linear-gradient(90deg, #7c3aed, #2563eb);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .hero p { color: #475569; font-size: 1.05rem; }
    .result { border-radius: 16px; padding: 1.2rem 1.4rem; margin-top: 1rem; color: #0f172a;
              box-shadow: 0 4px 14px rgba(79,70,229,.12); }
    .result.positive { background:#ecfdf5; border-left: 8px solid #10b981; }
    .result.negative { background:#fef2f2; border-left: 8px solid #ef4444; }
    .result h2 { margin: 0 0 .2rem 0; }
    .stButton > button { border-radius: 10px; border: 1px solid #c7d2fe; }
    .stButton > button[kind="primary"] { background: linear-gradient(90deg,#7c3aed,#2563eb);
               color: white; border: none; font-weight: 600; }
    .footer { text-align:center; color:#64748b; font-size:.85rem; margin-top:2.5rem; }
    </style>
    <div class="hero">
      <h1>💬 SentimentAI</h1>
      <p>Type a sentence and a machine learning model will tell you whether it sounds
      <b>positive</b> or <b>negative</b>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "user_text" not in st.session_state:
    st.session_state.user_text = ""
if "result" not in st.session_state:
    st.session_state.result = None


def set_example(sentence: str) -> None:
    st.session_state.user_text = sentence
    st.session_state.result = None


def clear_all() -> None:
    st.session_state.user_text = ""
    st.session_state.result = None


st.text_area("Your text", key="user_text", height=150, max_chars=MAX_CHARS,
             placeholder="e.g. I absolutely loved the service today!")

col_analyze, col_clear, _ = st.columns([2, 1, 3])
analyze = col_analyze.button("Analyze Sentiment", type="primary", use_container_width=True)
col_clear.button("Clear", on_click=clear_all, use_container_width=True)

st.caption("Try an example:")
example_cols = st.columns(2)
for i, sentence in enumerate(EXAMPLES):
    example_cols[i % 2].button(sentence, key=f"ex{i}", on_click=set_example,
                               args=(sentence,), use_container_width=True)

if analyze:
    text = st.session_state.user_text.strip()
    if not text:
        st.session_state.result = None
        st.warning("Please type or select some text first.")
    else:
        try:
            with st.spinner("Analyzing... (the first request can be slow if the server was asleep)"):
                st.session_state.result = predict_sentiment(text, API_URL, API_TIMEOUT)
        except ApiError as exc:
            st.session_state.result = None
            st.error(str(exc))

result = st.session_state.result
if result:
    label = result["sentiment"]
    icon = "😊" if label == "positive" else "😞"
    st.markdown(
        f'<div class="result {label}"><h2>{icon} {label.capitalize()}</h2>'
        f'<div>Confidence: <b>{result["confidence"] * 100:.1f}%</b></div></div>',
        unsafe_allow_html=True,
    )
    st.progress(float(result["confidence"]))
    with st.expander("See class probabilities"):
        for name, prob in result.get("probabilities", {}).items():
            st.write(f"**{name.capitalize()}**: {prob * 100:.1f}%")

with st.expander("How does the prediction work?"):
    st.markdown(
        """
1. **Clean** – your text is lowercased and stripped of links, symbols and numbers.
2. **Vectorize** – a *TF-IDF vectorizer* turns the words (and word pairs) into numbers.
3. **Classify** – a *Logistic Regression* model, trained on labeled examples, gives a probability for each sentiment.
4. **Answer** – the higher probability wins and is shown as the confidence score.

The model runs in a separate **FastAPI** service; this website only sends your text to it.
        """
    )

st.markdown(
    '<div class="footer">SentimentAI · Machine Learning Deployment Capstone Project</div>',
    unsafe_allow_html=True,
)
