import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import numpy as np
from pathlib import Path

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384
NUM_CLASSES = 3
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3

INTENT_TO_LABEL = {0: "symptom", 1: "drug_info", 2: "analogs"}
LABEL_TO_INTENT = {"symptom": 0, "drug_info": 1, "analogs": 2}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {DEVICE}")

class DrugQuestionDataset(Dataset):
    def __init__(self, texts, labels, model_name=MODEL_NAME):
        self.texts = texts
        self.labels = labels
        self.encoder = SentenceTransformer(model_name)
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Генерируем эмбеддинг
        embedding = self.encoder.encode(text, convert_to_numpy=True)
        
        return {
            'embedding': torch.FloatTensor(embedding),
            'label': torch.LongTensor([label])[0]
        }

class DrugQuestionClassifier(nn.Module):

    def __init__(self, input_dim=EMBEDDING_DIM, hidden_dim=128, num_classes=NUM_CLASSES):
        super(DrugQuestionClassifier, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

def load_dataset():
    print("\nЗагрузка датасетов...")
    
    train_path = Path("../data/dataset_train.json")
    val_path = Path("../data/dataset_val.json")
    test_path = Path("../data/dataset_test.json")
    
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    with open(val_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # Извлекаем тексты и метки
    train_texts = [item['text'] for item in train_data]
    train_labels = [LABEL_TO_INTENT[item['intent']] for item in train_data]
    
    val_texts = [item['text'] for item in val_data]
    val_labels = [LABEL_TO_INTENT[item['intent']] for item in val_data]
    
    test_texts = [item['text'] for item in test_data]
    test_labels = [LABEL_TO_INTENT[item['intent']] for item in test_data]
    
    print(f"  Train: {len(train_texts)} примеров")
    print(f"  Val:   {len(val_texts)} примеров")
    print(f"  Test:  {len(test_texts)} примеров")
    
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


def train_model():
    print("ОБУЧЕНИЕ НЕЙРОСЕТИ ДЛЯ КЛАССИФИКАЦИИ ВОПРОСОВ")
    
    # Загрузка данных
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_dataset()
    
    # Создаём даталоадеры
    print("\nСоздание даталоадеров...")
    
    # Для train создаём эмбеддинги заранее (быстрее)
    print("Кодирование текстов...")
    encoder = SentenceTransformer(MODEL_NAME)
    
    train_embeddings = encoder.encode(train_texts, show_progress_bar=True)
    val_embeddings = encoder.encode(val_texts, show_progress_bar=True)
    test_embeddings = encoder.encode(test_texts, show_progress_bar=True)
    
    # Конвертируем в тензоры
    class SimpleDataset(Dataset):
        def __init__(self, embeddings, labels):
            self.embeddings = torch.FloatTensor(embeddings)
            self.labels = torch.LongTensor(labels)
        
        def __len__(self):
            return len(self.embeddings)
        
        def __getitem__(self, idx):
            return self.embeddings[idx], self.labels[idx]
    
    train_dataset = SimpleDataset(train_embeddings, train_labels)
    val_dataset = SimpleDataset(val_embeddings, val_labels)
    test_dataset = SimpleDataset(test_embeddings, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Создаём модель
    print("\nСоздание модели...")
    model = DrugQuestionClassifier().to(DEVICE)
    print(f"  Параметры: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    
    # Обучение
    best_val_acc = 0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_embeddings, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            batch_embeddings = batch_embeddings.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            
            # Forward
            outputs = model(batch_embeddings)
            loss = criterion(outputs, batch_labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
        
        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        
        with torch.no_grad():
            for batch_embeddings, batch_labels in val_loader:
                batch_embeddings = batch_embeddings.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)
                
                outputs = model(batch_embeddings)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()
        
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Сохраняем лучшую модель
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"Лучшая модель сохранена!")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= 4:
            print(f"\n  Early stopping на эпохе {epoch+1}")
            break
    
    # Загружаем лучшую модель
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"\n  Загружена лучшая модель (Val Acc: {best_val_acc:.4f})")

    print("Тестирование")
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_embeddings, batch_labels in test_loader:
            batch_embeddings = batch_embeddings.to(DEVICE)
            
            outputs = model(batch_embeddings)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.numpy())
    
    # Метрики
    test_acc = accuracy_score(all_labels, all_preds)
    print(f"\nТочность на тесте: {test_acc:.4f}")
    
    print("\nОтчёт по классам:")
    target_names = ["symptom", "drug_info", "analogs"]
    report = classification_report(all_labels, all_preds, target_names=target_names)
    print(report)
    
    # Сохраняем модель
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': EMBEDDING_DIM,
            'hidden_dim': 128,
            'num_classes': NUM_CLASSES
        },
        'accuracy': test_acc
    }, '../models/drug_intent_classifier.pth')
    
    print(f"Модель сохранена: drug_intent_classifier.pth")
    print(f"Точность: {test_acc:.4f}")
    
    return model, test_acc

if __name__ == "__main__":
    model, acc = train_model()
    print("Обучение завершено!")
    print(f"\nФайлы:")
    print(f"  - drug_intent_classifier.pth (модель)")
    print(f"  - dataset_train.json, dataset_val.json, dataset_test.json (данные)")
