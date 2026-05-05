# FastAPI для RAG-системы лекарств

## Запуск

```bash
.venv\Scripts\python.exe api.py
```

Сервер запущен на `http://localhost:8000`

## Документация

После запуска откройте:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Эндпоинты

### 1. GET `/api/search?q=...&top_k=10`
Поиск препаратов по запросу

**Параметры:**
- `q` (string, обязательный) — поисковый запрос
- `top_k` (int, optional, default=10) — количество результатов

**Пример:**
```bash
curl "http://localhost:8000/api/search?q=парацетамол&top_k=5"
```

**Ответ:**
```json
[
  {
    "id": 123,
    "trade_name": "Парацетамол",
    "source_url": "https://www.vidal.ru/drugs/paracetamol",
    "manufacturer": "...",
    "description": "...",
    "indications": "...",
    "rank": 0.95
  }
]
```

---

### 2. GET `/api/drugs/{id}`
Полная информация о препарате

**Параметры:**
- `id` (int, обязательный) — ID препарата

**Пример:**
```bash
curl http://localhost:8000/api/drugs/123
```

**Ответ:**
```json
{
  "id": 123,
  "trade_name": "Парацетамол",
  "source_url": "...",
  "manufacturer": "...",
  "reg_number": "...",
  "substances": [{"name_ru": "парацетамол", "name_en": "paracetamol"}],
  "forms": [{"dosage": "500 мг", "package_size": "20 шт"}],
  "description": "...",
  "indications": "...",
  "contraindications": "...",
  "side_effects": "...",
  "usage_instructions": "..."
}
```

---

### 3. GET `/api/drugs/{id}/analogs?top_k=10&analog_type=all`
Поиск аналогов препарата

**Параметры:**
- `id` (int, обязательный) — ID препарата
- `top_k` (int, optional, default=10) — количество аналогов
- `analog_type` (string, optional, default="all") — тип:
  - `structural` — по активным веществам
  - `therapeutic` — по терапевтическому действию
  - `all` — оба типа

**Пример:**
```bash
curl "http://localhost:8000/api/drugs/123/analogs?top_k=5&analog_type=structural"
```

**Ответ:**
```json
{
  "structural": [
    {
      "id": 456,
      "trade_name": "Панадол",
      "source_url": "...",
      "substances": ["парацетамол"],
      "common_count": 1
    }
  ],
  "therapeutic": [...]
}
```

---

### 4. POST `/api/chat`
RAG чат — генерация ответа на вопрос

**Тело запроса:**
```json
{
  "query": "Что принимать при головной боли?",
  "top_k": 5
}
```

**Пример:**
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Что принимать при головной боли?", "top_k": 5}'
```

**Ответ:**
```json
{
  "answer": "Найдено 5 препаратов:\n\n1. **Парацетамол**\n   Показания: ...\n...",
  "sources": [...],
  "type": "qa_match"
}
```

---

### 5. GET `/api/stats`
Статистика по базе данных

**Пример:**
```bash
curl http://localhost:8000/api/stats
```

**Ответ:**
```json
{
  "total_drugs": 14859,
  "total_substances": 1597,
  "total_content": 14859
}
```

---

## Примеры использования (Python)

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Поиск
response = requests.get(f"{BASE_URL}/api/search", params={
    "q": "ибупрофен",
    "top_k": 5
})
drugs = response.json()
print(f"Найдено: {len(drugs)} препаратов")

# 2. Информация о препарате
drug_id = drugs[0]["id"]
response = requests.get(f"{BASE_URL}/api/drugs/{drug_id}")
details = response.json()
print(f"Название: {details['trade_name']}")
print(f"Вещества: {details['substances']}")

# 3. Аналоги
response = requests.get(f"{BASE_URL}/api/drugs/{drug_id}/analogs", params={
    "top_k": 5,
    "analog_type": "structural"
})
analogs = response.json()
print(f"Структурные аналоги: {len(analogs['structural'])}")

# 4. RAG чат
response = requests.post(f"{BASE_URL}/api/chat", json={
    "query": "Что принимать при температуре?",
    "top_k": 5
})
chat = response.json()
print(chat["answer"])

# 5. Статистика
response = requests.get(f"{BASE_URL}/api/stats")
stats = response.json()
print(f"Всего препаратов: {stats['total_drugs']}")
```
