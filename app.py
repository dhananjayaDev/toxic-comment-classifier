import streamlit as st
from utils import load_model, predict_labels

st.set_page_config(page_title="Toxic Comment Classifier", layout="centered")
st.title("Multi-Label Toxic Comment Classifier")
st.markdown("Classify comments into multiple toxicity categories using a fine-tuned BERT model.")

# Load model and tokenizer
@st.cache_resource
def get_model():
    return load_model()

tokenizer, model = get_model()

# Input text
text_input = st.text_area("Enter a comment:", height=150)

if st.button("Classify"):
    if text_input.strip():
        with st.spinner("Analyzing..."):
            predictions = predict_labels(text_input, tokenizer, model)
        if predictions:
            st.success("🚨 Toxic traits detected:")
            for label, score in predictions.items():
                st.write(f"- **{label}**: {score:.2f}")
        else:
            st.success("✅ No toxicity detected.")
    else:
        st.warning("Please enter a comment to classify.")