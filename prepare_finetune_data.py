"""
prepare_finetune_data.py
Генерация датасета для Fine-Tuning Qwen2.5-1.5B
Формат: JSONL (ChatML / Alpaca style)
"""
import json
import random
import psycopg2
from tqdm import tqdm

DB_CONFIG = {
    "dbname": "medicines_db",
    "user": "postgres",
    "password": "bkmzrjh1231",
    "host": "localhost",
    "port": "5433"
}


def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)


def generate_finetune_dataset(output_file="finetune_dataset.jsonl", limit=2000):
    """
    Генерирует датасет для обучения LLM.
    Структура: {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
    """
    conn = get_db_connection()
    cur = conn.cursor()

    dataset = []

    print("Извлечение данных из БД...")
    # Берем препараты с хорошим описанием
    cur.execute("""
        SELECT d.id, d.trade_name, d.manufacturer, 
               dc.description, dc.indications, dc.contraindications, 
               dc.usage_instructions, dc.side_effects
        FROM drugs d
        JOIN drug_content dc ON d.id = dc.drug_id
        WHERE dc.description IS NOT NULL AND length(dc.description) > 50
        LIMIT %s;
    """, (limit,))

    drugs = cur.fetchall()

    print(f"Генерация примеров для {len(drugs)} препаратов...")

    for drug in tqdm(drugs):
        drug_id, name, manufacturer, desc, indications, contraind, usage, side_effects = drug

        # 1. Вопрос про инструкцию
        q1 = f"Расскажи подробно о препарате {name}."
        a1 = f"💊 **{name}**\n"
        if manufacturer: a1 += f"Производитель: {manufacturer}\n"
        if desc: a1 += f"\n📝 **Описание:** {desc[:500]}\n"
        if indications: a1 += f"\n✅ **Показания:** {indications[:400]}\n"
        if contraind: a1 += f"\n⛔ **Противопоказания:** {contraind[:300]}\n"
        if usage: a1 += f"\n💊 **Применение:** {usage[:400]}\n"
        if side_effects: a1 += f"\n⚠️ **Побочные эффекты:** {side_effects[:300]}\n"

        dataset.append({
            "messages": [
                {"role": "system",
                 "content": "Ты полезный медицинский ассистент. Отвечай точно и вежливо, используя только предоставленные факты."},
                {"role": "user", "content": q1},
                {"role": "assistant", "content": a1.strip()}
            ]
        })

        # 2. Вопрос про показания (если есть)
        if indications and len(indications) > 20:
            q2 = f"От чего помогает {name}?"
            a2 = f"Препарат {name} применяется при следующих состояниях:\n{indications[:500]}"
            dataset.append({
                "messages": [
                    {"role": "system", "content": "Ты медицинский справочник. Кратко отвечай на вопросы о лекарствах."},
                    {"role": "user", "content": q2},
                    {"role": "assistant", "content": a2.strip()}
                ]
            })

        # 3. Вопрос про противопоказания (если есть)
        if contraind and len(contraind) > 20:
            q3 = f"Какие противопоказания у {name}?"
            a3 = f"Противопоказания к применению {name}:\n{contraind[:500]}"
            dataset.append({
                "messages": [
                    {"role": "system", "content": "Ты врач-консультант. Предупреждай о рисках."},
                    {"role": "user", "content": q3},
                    {"role": "assistant", "content": a3.strip()}
                ]
            })

    # Сохранение в JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n✓ Датасет сохранен в {output_file}")
    print(f"  Всего примеров: {len(dataset)}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    generate_finetune_dataset()