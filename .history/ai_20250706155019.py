import torch
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import numpy as np

# ─── DATA ──────────────────────────────────────────────
conversations = [
    ("hi", "hello!"),
    ("hello", "hey there!"),
    ("how are you?", "i'm doing great, thanks!"),
    ("what's your name?", "i'm a chatbot created by Wahb."),
    ("what do you do?", "i chat with humans and learn."),
    ("bye", "goodbye!"),
    ("see you", "take care!"),
    ("what is ai?", "ai stands for artificial intelligence."),
    ("how does ai work?", "it uses algorithms to learn from data."),
    ("what is pytorch?", "it's a machine learning library."),
    ("who made you?", "Wahb did!"),
    ("do you like humans?", "absolutely! humans are fascinating."),
    ("can you learn?", "yes, with enough data and training."),
    ("tell me a joke", "why did the AI break up? it lost its logic."),
    ("what is your purpose?", "to assist and chat with people like you!"),
    ("thanks", "you're welcome!"),
    ("thank you", "no problem!"),
    ("are you smart?", "i'm getting there, thanks to training!"),
    ("how old are you?", "i was born today."),
    ("who is your creator?", "my creator is Wahb, the legend."),
] * 100  # 🔥 20 * 100 = 2000 samples

# ─── TOKENIZER ──────────────────────────────────────────
class Tokenizer:
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.idx2word = ["<pad>", "<unk>"]

    def fit(self, sentences):
        for sentence in sentences:
            for word in sentence.lower().split():
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.idx2word)
                    self.idx2word.append(word)

    def encode(self, sentence, max_len=10):
        tokens = [self.word2idx.get(word, 1) for word in sentence.lower().split()]
        tokens = tokens[:max_len] + [0] * (max_len - len(tokens))
        return tokens

    def decode(self, tokens):
        return " ".join([self.idx2word[token] for token in tokens if token != 0])

# ─── DATASET ────────────────────────────────────────────
class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ─── MODEL ──────────────────────────────────────────────
class ChatBot(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        _, h = self.rnn(x)
        out = self.fc(h.squeeze(0))
        return out

# ─── TRAINING ───────────────────────────────────────────
def train_model(model, data_loader, optimizer, criterion, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"🧠 Epoch {epoch+1:02d} | Loss: {total_loss/len(data_loader):.4f}")

# ─── MAIN ───────────────────────────────────────────────
if __name__ == "__main__":
    tokenizer = Tokenizer()
    tokenizer.fit([q for q, _ in conversations] + [a for _, a in conversations])

    X = [tokenizer.encode(q) for q, _ in conversations]
    y = [tokenizer.encode(a)[0] for _, a in conversations]  # Predict just the first word of response

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

    train_ds = ChatDataset(X_train, y_train)
    val_ds = ChatDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    vocab_size = len(tokenizer.idx2word)
    model = ChatBot(vocab_size)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    train_model(model, train_loader, optimizer, criterion)

    # 🧪 Test it
    model.eval()
    while True:
        msg = input("👤 You: ")
        if msg.lower() in ["quit", "exit"]:
            break
        encoded = torch.tensor([tokenizer.encode(msg)], dtype=torch.long)
        with torch.no_grad():
            out = model(encoded)
            pred_token = out.argmax(1).item()
            print("🤖 Bot:", tokenizer.decode([pred_token]))
