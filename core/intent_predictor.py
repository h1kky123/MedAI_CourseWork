import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from typing import Dict, Optional

MODEL_PATH = "models/drug_intent_classifier.pth"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

INTENT_LABELS = {0: "symptom", 1: "drug_info", 2: "analogs"}
LABEL_INTENTS = {"symptom": 0, "drug_info": 1, "analogs": 2}


class DrugQuestionClassifier(nn.Module):
    # Архитектура классификатора намерений
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


class IntentPredictor:
    def __init__(self, model_path=MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = DrugQuestionClassifier()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        self.accuracy = checkpoint.get('accuracy', 0)

    def predict(self, question: str) -> Dict:
        # Классификация намерения и извлечение сущности из вопроса
        embedding = self.encoder.encode(question, convert_to_numpy=True)
        embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(embedding_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            predicted_class = torch.argmax(output, dim=1).item()

        intent = INTENT_LABELS[predicted_class]
        confidence = probabilities[predicted_class].item()
        entity = self._extract_entity(question, intent)

        return {'intent': intent, 'confidence': confidence, 'entity': entity}

    def _extract_entity(self, question: str, intent: str) -> Optional[str]:
        # Извлечение названия препарата или симптома из текста вопроса
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
            entity_words = []
            for word in words:
                word_clean = "".join(c for c in word if c.isalpha())
                if len(word_clean) > 3 and word_clean.lower() not in stop_words:
                    entity_words.append(word_clean)
            return " ".join(entity_words) if entity_words else None

        return None