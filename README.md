# MedAI — система для лекарственных препаратов

Веб-приложение для поиска информации о лекарственных препаратах, поиска аналогов и ответов на медицинские вопросы. Система построена на основе нейросетей и базы данных из 14 859 препаратов.
## 🧠 Архитектура

Система состоит из трёх нейросетевых компонентов:

- **DrugQuestionClassifier** — собственная нейросеть на PyTorch для классификации намерений пользователя (симптом / информация о препарате / поиск аналогов)
- **Qwen2.5-1.5B-Instruct + LoRA** — дообученная языковая модель для генерации ответов
- **paraphrase-multilingual-MiniLM-L12-v2** — модель эмбеддингов для семантического поиска

## 🛠 Технологии

- **PostgreSQL + pgvector** — база данных с полнотекстовым и векторным поиском
- **FastAPI** — REST API бэкенд
- **PyTorch** — обучение и инференс нейросетей
- **PEFT / LoRA** — дообучение языковой модели
- **sentence-transformers** — векторные эмбеддинги

## 📁 Структура проекта

```
CourseWork/
├── core/                        # Основные модули приложения
│   ├── api.py                   # FastAPI приложение
│   ├── vector_store.py          # Работа с PostgreSQL и векторный поиск
│   ├── drug_analogs.py          # Поиск аналогов препаратов
│   └── intent_predictor.py      # Классификатор намерений
│
├── ml/                          # Обучение моделей
│   ├── train_classifier.py      # Обучение классификатора намерений
│   ├── train_lora_qwen.py       # Дообучение Qwen + LoRA
│   ├── generate_embeddings.py   # Генерация векторных эмбеддингов
│   ├── convert_embeddings.py    # Конвертация эмбеддингов
│   └── prepare_finetune_data.py # Подготовка датасета для файнтюна
│
├── data/                        # Датасеты
│   ├── dataset_train.json
│   ├── dataset_val.json
│   └── dataset_test.json
│
├── scripts/                     # Вспомогательные скрипты
│   ├── parser_data.py           # Парсер vidal.ru
│   ├── data_cleaner.py          # Очистка данных
│   ├── generate_dataset.py      # Генерация датасета
│   └── load_dataset_to_db.py    # Загрузка данных в БД
│
├── models/                      # Модели (не включены в репозиторий)
│   ├── qwen_medical_lora/       # LoRA адаптеры
│   └── drug_intent_classifier.pth
│
├── static/                      # Фронтенд
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 Установка и запуск

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Настройка базы данных

Убедитесь что PostgreSQL запущен и настроен. Параметры подключения в `core/vector_store.py`:

```python
DB_CONFIG = {
    "dbname": "medicines_db",
    "user": "postgres",
    "password": "your_password",
    "host": "localhost",
    "port": "5433"
}
```

### 3. Запуск приложения

```bash
python -m core.api
```

Интерфейс: http://localhost:8000  
Документация API: http://localhost:8000/docs

## 📡 API

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/` | Веб-интерфейс |
| POST | `/api/chat` | RAG чат |

### Пример запроса

```python
import requests

r = requests.post("http://localhost:8000/api/chat", json={
    "query": "Что принимать при головной боли?",
    "top_k": 5
})
print(r.json())
```

### Типы ответов

| Тип | Описание |
|-----|----------|
| `drug_info` | Информация о конкретном препарате |
| `analogs` | Список аналогов с комментарием ИИ |
| `symptom` | Подбор препаратов по симптому |
| `selection` | Уточнение при неоднозначном запросе |

## 🗃 База данных

```
drugs ─────┬── drug_content     (описание, показания, эмбеддинги)
           ├── drug_forms        (формы выпуска)
           └── drug_substances ──active_substances (активные вещества)
```

| Таблица | Записей |
|---------|---------|
| drugs | 14 859 |
| active_substances | 1 597 |
| drug_substances | 15 470 |
| drug_content | 14 859 |

## ⚙️ Обучение моделей

```bash
# Генерация датасета
python scripts/generate_dataset.py

# Обучение классификатора намерений
python ml/train_classifier.py

# Дообучение Qwen + LoRA
python ml/train_lora_qwen.py

# Генерация эмбеддингов для векторного поиска
python ml/generate_embeddings.py
```

## 📊 Повторный сбор данных

```bash
python scripts/parser_data.py   
python scripts/data_cleaner.py 
python scripts/load_dataset_to_db.py
```

---

**Версия:** 5.0 | **Дата:** Май 2026