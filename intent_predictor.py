"""
Модуль предсказания намерений пользователя
Использует обученную нейросеть для классификации вопросов
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from typing import Dict, Optional

# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================

MODEL_PATH = "drug_intent_classifier.pth"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

INTENT_LABELS = {0: "symptom", 1: "drug_info", 2: "analogs"}
LABEL_INTENTS = {"symptom": 0, "drug_info": 1, "analogs": 2}


# ============================================================
# НЕЙРОСЕТЬ (такая же архитектура как при обучении)
# ============================================================

class DrugQuestionClassifier(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=128, num_classes=3):
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


# ============================================================
# КЛАСС ПРЕДСКАЗАНИЯ
# ============================================================

class IntentPredictor:
    """Предсказывает намерение пользователя по вопросу"""
    
    def __init__(self, model_path=MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Загружаем эмбеддер
        print("Загрузка эмбеддера...")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)
        
        # Загружаем модель
        print(f"Загрузка модели из {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.model = DrugQuestionClassifier()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.accuracy = checkpoint.get('accuracy', 0)
        print(f"✓ Модель загружена (точность: {self.accuracy:.4f})")
    
    def predict(self, question: str) -> Dict:
        """
        Предсказывает намерение и извлекает сущность
        
        Returns:
            {
                'intent': 'symptom' | 'drug_info' | 'analogs',
                'confidence': 0.95,
                'entity': 'Нурофен' | None,
                'entity_id': 123 | None
            }
        """
        
        # Генерируем эмбеддинг
        embedding = self.encoder.encode(question, convert_to_numpy=True)
        embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(self.device)
        
        # Предсказание
        with torch.no_grad():
            output = self.model(embedding_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            predicted_class = torch.argmax(output, dim=1).item()
        
        intent = INTENT_LABELS[predicted_class]
        confidence = probabilities[predicted_class].item()
        
        # Извлекаем сущность (название препарата или симптом)
        entity = self._extract_entity(question, intent)
        
        return {
            'intent': intent,
            'confidence': confidence,
            'entity': entity
        }

    def _extract_entity(self, question: str, intent: str) -> Optional[str]:
        """Извлекает название препарата или симптом из вопроса"""
        words = question.split()

        if intent == "symptom":
            symptom_keywords = ["голов", "живот", "горло", "температур", "кашл",
                                "насморк", "зуб", "спин", "тошнот", "давлен",
                                "аллерг", "бессонн", "диар", "запор", "ангин",
                                "грипп", "воспален", "боль"]

            for word in words:
                word_lower = word.lower()
                if any(kw in word_lower for kw in symptom_keywords):
                    return word_lower

            return " ".join(words[-2:]) if len(words) > 2 else question

        elif intent in ["drug_info", "analogs"]:
            stop_words = {"что", "как", "какие", "какой", "инструкция", "применение",
                          "аналоги", "аналог", "заменить", "заменители", "похожие",
                          "препарат", "лекарство", "таблетки", "расскажи", "расскажите",
                          "показания", "противопоказания", "использовать", "это", "для",
                          "при", "у", "есть", "ли", "и", "или"}

            # Собираем ВСЕ подходящие слова, а не только первое
            entity_words = []
            for word in words:
                word_clean = "".join(c for c in word if c.isalpha())
                if len(word_clean) > 3 and word_clean.lower() not in stop_words:
                    entity_words.append(word_clean)  # ← накапливаем

            # Возвращаем все слова как одну строку
            return " ".join(entity_words) if entity_words else None

        return None


# ============================================================
# ТЕСТИРОВАНИЕ
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("ТЕСТИРОВАНИЕ ПРЕДСКАЗАТЕЛЯ НАМЕРЕНИЙ")
    print("="*60)
    
    predictor = IntentPredictor()
    
    # Тестовые вопросы
    test_questions = [
        "Какие аналоги у Нурофена?",
        "Что делать при головной боли?",
        "Инструкция для Парацетамола",
        "Чем заменить Ибупрофен?",
        "Что принимать при температуре?",
        "Показания для Аспирина",
        "Есть ли аналоги у Амoxicillin?",
        "Что помогает от кашля?",
    ]
    
    print(f"\n{'='*60}")
    print("ПРЕДСКАЗАНИЯ:")
    print(f"{'='*60}")
    
    for question in test_questions:
        result = predictor.predict(question)
        
        print(f"\nВопрос: {question}")
        print(f"  Намерение: {result['intent']}")
        print(f"  Уверенность: {result['confidence']:.4f}")
        print(f"  Сущность: {result['entity']}")
