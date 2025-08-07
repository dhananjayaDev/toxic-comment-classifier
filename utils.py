import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Toxicity labels
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Load tokenizer and model
def load_model(model_path='model', tokenizer_path='model'):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

# Predict toxicity labels
def predict_labels(text, tokenizer, model, threshold=0.5):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    predictions = {label: float(prob) for label, prob in zip(LABELS, probs)}
    filtered = {k: v for k, v in predictions.items() if v > threshold}
    return filtered