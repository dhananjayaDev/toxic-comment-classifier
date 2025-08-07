#  Toxic Comment Classifier

![Banner](toxic_comment_classifier.png)

A multi-label toxicity detection app built with a fine-tuned BERT model and Streamlit. This tool classifies user comments into six toxicity categories and provides confidence scores for each label.

---

##  Features

- Multi-label classification using BERT
- Streamlit-powered interactive UI
- Confidence thresholding
- Clean modular structure (`app.py`, `utils.py`)
- Easy to extend and deploy
- Recruiter-friendly layout and UX

---

##  Toxicity Labels

The model predicts the following categories:

- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

---

##  Tech Stack

- [Transformers](https://huggingface.co/transformers/)
- [PyTorch](https://pytorch.org/)
- [Streamlit](https://streamlit.io/)
- BERT (`bert-base-uncased` fine-tuned)

---

## Installation

```bash
git clone https://github.com/dhananjayaDev/toxic-comment-classifier.git
cd toxic-comment-classifier
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

##  Run the App

```bash
streamlit run app.py
```

Then open your browser to: [http://localhost:8501](http://localhost:8501)

---

##  Sample Comments

```text
You're such a pathetic loser. No one wants you around.
```

Expected output:
- `toxic`
- `insult`
- `severe_toxic` (if trained)

```text
I disagree with your opinion, but I respect your right to express it.
```

Expected output:
- ✅ No toxicity detected

---

##  Folder Structure

```
toxic-comment-classifier/
├── app.py
├── utils.py
├── requirements.txt
├── README.md
├── .gitignore
├── model/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   ├── vocab.txt
└── venv/
```

---

##  Model Details

- Fine-tuned on a multi-label toxicity dataset
- Uses sigmoid activation for independent label probabilities
- Saved using Hugging Face's `save_pretrained()` format
- Thresholding logic handled in `utils.py`

---

##  UI/UX Notes

- Compact layout with confidence bars
- Threshold slider for label sensitivity
- Responsive design for recruiter demos
- Easily customizable with Streamlit widgets

---

##  Deployment Tips

- Use [Streamlit Community Cloud](https://streamlit.io/cloud) for free hosting
- Or deploy via [Hugging Face Spaces](https://huggingface.co/spaces)
- Include a demo video or screenshots in your README for extra polish

---

##  Author

**Dhananjaya**  
[GitHub Profile](https://github.com/dhananjayaDev)

---

##  License

This project is licensed under the MIT License.

