import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import spacy
import re
import pickle

with open("Encoder/ENCODER.pkl", "rb") as f:
    encoder = pickle.load(f)

def preprocess_text(text):
    nlp = spacy.load("en_core_web_sm")
    text = re.sub(r'[\n\t\r]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    doc = nlp(text)
    clean_tokens = [token.text for token in doc if not token.is_punct]
    clean_text = ' '.join(clean_tokens)

    return clean_text

class ResumeClassifier(nn.Module):
    def __init__(self, n_labels):
        super().__init__()

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        for layer in self.bert.parameters():
            layer.requires_grad = False

        self.fc = nn.Linear(768, n_labels)
        self.act = nn.Softmax(dim=-1)

        self.to(self.device)

    def forward(self, sents):
        tokens = self.tokenizer(sents, return_tensors='pt', truncation=True, padding=True).to(self.device)
        cls_token = self.bert(**tokens).last_hidden_state[:, 0, :]
        preds = self.fc(cls_token)
        return preds

    def predict(self, sents):
        preds = self.forward(sents)
        return self.act(preds)

def create_classifier(num_classes):
    model = ResumeClassifier(num_classes)
    try:
        model.load_state_dict(torch.load("BERT/BERT.pt"))
        print("Pretrained model loaded successfully")
    except Exception:
        print("Number is different from model's. Using not pretrained version")
    return model

def make_prediction(model, text):
    label = torch.argmax(model.predict(text), dim=-1).data.cpu()
    result = encoder.inverse_transform(label)
    return result[0]


