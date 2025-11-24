import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim import AdamW


# data processing
csv_path = './enhanced_imdb_data.csv'
df = pd.read_csv(csv_path, nrows=5000)
# df = pd.read_csv(csv_path)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    return text

df = df.dropna(subset=['sentiment'])
df['review'] = df['review'].apply(clean_text)
df['sentiment'] = df['sentiment'].astype(str)
unique_labels = sorted(df['sentiment'].unique())
label2id = {lab: i for i, lab in enumerate(unique_labels)}
id2label = {i: lab for lab, i in label2id.items()}
df['label'] = df['sentiment'].map(label2id)

train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Dataset and DataLoader
class ReviewDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.texts = df["review"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = ReviewDataset(train_df, tokenizer)
valid_dataset = ReviewDataset(valid_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8)

# BERT 
class BERTSentimentClassifier(nn.Module):
    def __init__(self, n_classes=len(unique_labels), dropout_prob=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        last_hidden = outputs.last_hidden_state  # (B, L, H)
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()

        mean_pool = (last_hidden * mask).sum(1) / mask.sum(1)
        x = self.dropout(mean_pool)
        return self.linear(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTSentimentClassifier().to(device)

for name, param in model.bert.named_parameters():
    if name.startswith("encoder.layer.") and int(name.split(".")[2]) < 6:
        param.requires_grad = False

# Optimizer and Scheduler
EPOCHS = 10
LEARNING_RATE = 1e-5

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
criterion = nn.CrossEntropyLoss()

# Training and validation functions
def train_one_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    acc = correct / len(loader.dataset)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    return total_loss / len(loader), acc, precision, recall, f1

history = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "val_acc": [],
    "precision": [],
    "recall": [],
    "f1": []
}

# Primary Training Cycle
for epoch in range(EPOCHS):
    train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion)
    val_loss, val_acc, precision, recall, f1 = evaluate(model, valid_loader, criterion)
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Valid Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {f1:.4f}")

    history["epoch"].append(epoch + 1)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["precision"].append(precision)
    history["recall"].append(recall)
    history["f1"].append(f1)

df_history = pd.DataFrame(history)
df_history.to_csv("bert_training_log.csv", index=False)
print("Training log saved to bert_training_log.csv")

# Attention 
model.eval()

text = "A very charming film with wonderful sentiment and heart"

# encode
inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=128
).to(device)

raw_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

with torch.no_grad():
    bert_outputs = model.bert(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        output_attentions=True
    )

attention_weights = bert_outputs.attentions  # list of 12 tensors

seq_len = (inputs['attention_mask'] > 0).sum().item()

# last layer
last_layer_attention = attention_weights[-1][0].mean(dim=0).cpu().numpy()

# ture lenth
last_layer_attention = last_layer_attention[:seq_len, :seq_len]
raw_tokens = raw_tokens[:seq_len]

# token
special_tokens = set(tokenizer.all_special_tokens)

valid_idx = [i for i, t in enumerate(raw_tokens) if t not in special_tokens]
tokens = [raw_tokens[i] for i in valid_idx]

# attention
att_map = last_layer_attention[np.ix_(valid_idx, valid_idx)]

# heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    att_map,
    xticklabels=tokens,
    yticklabels=tokens,
    cmap="viridis",
    linewidths=.5,
    linecolor='black'
)
plt.title(f"BERT Attention Map (Layer 12, Mean Heads)\nSentence: {text[:40]}...", fontsize=14)
plt.savefig("./bert_attention_map.png", dpi=300)
plt.show()
