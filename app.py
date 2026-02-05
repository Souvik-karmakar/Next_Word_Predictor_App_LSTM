import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page configuration
st.set_page_config(
    page_title="Next Word Prediction",
    page_icon="🧠",
    layout="centered"
)

# Load the LSTM model
model = load_model("next_word_lstm.h5")

# Load the tokenizer
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
        padding="pre"
    )

    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return "Not found"

# App UI
st.markdown(
    "<h1 style='text-align: center;'>🧠 Next Word Prediction</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Predict the next word using an LSTM-based Language Model</p>",
    unsafe_allow_html=True
)

st.divider()

# Input section
input_text = st.text_input(
    "Enter a sequence of words",
    value="To be or not to",
    help="Type a sentence and predict the next word"
)

col1, col2 = st.columns([1, 1])

with col1:
    predict_btn = st.button("🔮 Predict Next Word")

with col2:
    clear_btn = st.button("🧹 Clear")

if clear_btn:
    st.experimental_rerun()

# Prediction output
if predict_btn and input_text.strip():
    with st.spinner("Predicting..."):
        max_sequence_len = model.input_shape[1] + 1
        next_word = predict_next_word(
            model, tokenizer, input_text, max_sequence_len
        )

    st.success("Prediction Successful ✅")

    st.markdown(
        f"""
        <div style="
            background-color:#f0f2f6;
            padding:15px;
            border-radius:10px;
            font-size:18px;">
            <b>Input:</b> {input_text} <br><br>
            <b>Predicted Next Word:</b> <span style="color:#4CAF50;">{next_word}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

# Footer
st.markdown(
    "<p style='text-align:center; font-size:12px;'>Built with ❤️ using Streamlit & LSTM</p>",
    unsafe_allow_html=True
)
