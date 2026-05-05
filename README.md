# 💊 RAG-система для лекарственных препаратов

REST API для поиска, анализа и генерации ответов о лекарственных препаратах на основе данных vidal.ru

## 🛠 Технологии

- **PostgreSQL** — база данных с полнотекстовым и векторным поиском
- **sentence-transformers** (paraphrase-multilingual-MiniLM-L12-v2) — эмбеддинги
- **FastAPI** — REST API с автодокументацией
- **Qwen2.5-1.5B-Instruct** — генерация ответов (подключается позже)

## 📁 Структура проекта

```
CourseWork/
├── api.py                    # FastAPI приложение
├── parser_enhanced.py        # Парсер vidal.ru
├── vector_store_v2.py        # Работа с PostgreSQL
├── drug_analogs_v2.py        # Поиск аналогов
├── data_cleaner.py           # Очистка данных
├── load_dataset_to_db.py     # Загрузка JSON в БД
├── generate_embeddings.py    # Генерация эмбеддингов
├── requirements.txt          # Зависимости
├── API_GUIDE.md             # Документация API
├── database_schema.md       # Схема БД
└── README.md                # Этот файл
```

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
.venv\Scripts\pip.exe install -r requirements.txt
```

### 2. Запуск API

```bash
.venv\Scripts\python.exe api.py
```

Сервер: http://localhost:8000  
Документация: http://localhost:8000/docs

## 📡 API Эндпоинты

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/api/search?q=...` | Поиск препаратов |
| GET | `/api/drugs/{id}` | Информация о препарате |
| GET | `/api/drugs/{id}/analogs` | Поиск аналогов |
| POST | `/api/chat` | RAG чат |
| GET | `/api/stats` | Статистика |

### Примеры

```python
import requests

BASE = "http://localhost:8000"

# Поиск
r = requests.get(f"{BASE}/api/search", params={"q": "парацетамол", "top_k": 3})

# Информация
r = requests.get(f"{BASE}/api/drugs/123")

# Аналоги
r = requests.get(f"{BASE}/api/drugs/123/analogs", params={"analog_type": "structural"})

# Чат
r = requests.post(f"{BASE}/api/chat", json={"query": "Что при головной боли?", "top_k": 5})
```

Полная документация: [API_GUIDE.md](API_GUIDE.md)

## 🗃 База данных

### Схема

```
drugs ─────┬── drug_content (описание, показания, эмбеддинги)
           ├── drug_forms (формы выпуска)
           └── drug_substances ── active_substances (MNN)
```

### Статистика

| Таблица | Записей |
|---------|---------|
| drugs | 14,859 |
| active_substances | 1,597 |
| drug_substances | 15,470 |
| drug_forms | 13,345 |
| drug_content | 14,859 (с эмбеддингами) |

## 📊 Повторный парсинг

```bash
.venv\Scripts\python.exe parser_enhanced.py
```

Парсер собирает полную фарм-карточку:
- Активные вещества (MNN)
- Дозировки
- Производитель, рег. номер
- Описание, показания, противопоказания
- Побочные эффекты, взаимодействия

## ⚙️ Настройка БД

```sql
-- Подключение
psql -U postgres -h localhost -p 5433 -d medicines_db

-- Статистика
SELECT COUNT(*) FROM drugs;
SELECT COUNT(*) FROM active_substances;
SELECT COUNT(*) FROM drug_content WHERE embedding IS NOT NULL;
```

## 🎯 Следующие шаги

- [ ] Подключить Qwen2.5 к `/api/chat`
- [ ] Добавить кэширование ответов
- [ ] Создать веб-интерфейс

---

**Дата:** Апрель 2026  
**Версия:** 2.0 (FastAPI)
