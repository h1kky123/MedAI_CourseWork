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

SYMPTOM_TEMPLATES = [
    "Что делать при {symptom}?",
    "Что принимать при {symptom}?",
    "Какие лекарства помогают при {symptom}?",
    "Чем лечить {symptom}?",
    "Что помогает от {symptom}?",
    "Какое лекарство от {symptom}?",
    "Что пить если {symptom}?",
    "Подскажите препарат от {symptom}",
    "Посоветуйте лекарство при {symptom}",
]

DRUG_TEMPLATES = [
    "Что такое {drug_name}?",
    "Расскажи про {drug_name}",
    "Инструкция для {drug_name}",
    "Как применять {drug_name}?",
    "{drug_name} - что это за лекарство?",
    "Показания для {drug_name}",
    "Как использовать {drug_name}?",
    "Описание препарата {drug_name}",
    "Для чего назначают {drug_name}?",
    "{drug_name} противопоказания",
]

ANALOG_TEMPLATES = [
    "Какие аналоги у {drug_name}?",
    "Есть ли аналоги у {drug_name}?",
    "Чем заменить {drug_name}?",
    "Аналоги препарата {drug_name}",
    "Заменители {drug_name}",
    "Похожие препараты на {drug_name}",
    "{drug_name} аналоги",
]


def generate_dataset(num_symptoms=50, num_drugs=200, num_analogs=100):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    dataset = []

    # 1. Вопросы по симптомам
    symptoms_map = {
        "головная боль": ["головн", "голов"],
        "боль в животе": ["живот", "боли в живот"],
        "боль в горле": ["горл", "фарингит"],
        "температура": ["температур", "лихорадк", "жар"],
        "кашель": ["кашл", "бронхит"],
        "насморк": ["насморк", "ринит"],
        "зубная боль": ["зубн", "зуб"],
        "боль в спине": ["спин", "остеохондроз"],
        "тошнота": ["тошнот", "рвот"],
        "высокое давление": ["давлен", "гипертензи", "гипертон"],
        "боль в суставах": ["сустав", "артрит", "артроз"],
        "аллергия": ["аллерг", "аллергич"],
        "бессонница": ["бессонн", "сон", "инсомн"],
        "диарея": ["диар", "понос"],
        "запор": ["запор"],
        "ангина": ["ангин", "тонзиллит"],
        "грипп": ["грипп", "простуд", "орви"],
        "воспаление": ["воспален"],
    }

    for symptom, keywords in symptoms_map.items():
        all_tokens = []
        for kw in keywords:
            all_tokens.extend(kw.split())
        query_str = " | ".join(all_tokens[:5])

        cur.execute("""
            SELECT d.id, d.trade_name, d.manufacturer, dc.indications, dc.description
            FROM drugs d
            JOIN drug_content dc ON d.id = dc.drug_id
            WHERE dc.search_vector @@ to_tsquery('russian', %s)
            LIMIT 10;
        """, (query_str,))

        drugs_for_symptom = cur.fetchall()

        if drugs_for_symptom:
            for template in random.sample(SYMPTOM_TEMPLATES, min(5, len(SYMPTOM_TEMPLATES))):
                question = template.format(symptom=symptom)
                answer = f"При {symptom} могут помочь следующие препараты:\n\n"
                for i, drug in enumerate(drugs_for_symptom[:5], 1):
                    answer += f"{i}. {drug[1]}"
                    if drug[3]:
                        answer += f" — {drug[3][:150]}..."
                    answer += "\n"

                dataset.append({
                    "text": question,
                    "intent": "symptom",
                    "entity": symptom,
                    "answer": answer,
                    "sources": [{"id": d[0], "name": d[1]} for d in drugs_for_symptom[:5]]
                })

    # 2. Вопросы о конкретных препаратах
    cur.execute("""
        SELECT d.id, d.trade_name, d.manufacturer,
               dc.description, dc.indications, dc.contraindications,
               dc.usage_instructions, dc.side_effects
        FROM drugs d
        JOIN drug_content dc ON d.id = dc.drug_id
        WHERE dc.description IS NOT NULL OR dc.indications IS NOT NULL
        LIMIT 500;
    """)

    drugs = cur.fetchall()

    for drug in tqdm(drugs[:num_drugs], desc="Препараты"):
        drug_id, name, manufacturer, desc, indications, contraind, usage, side_effects = drug

        for template in random.sample(DRUG_TEMPLATES, random.randint(3, 5)):
            question = template.format(drug_name=name)
            answer = f"💊 **{name}**\n\n"
            if manufacturer:
                answer += f"Производитель: {manufacturer}\n\n"
            if desc:
                answer += f"Описание: {desc[:400]}\n\n"
            if indications:
                answer += f"Показания: {indications[:300]}\n\n"
            if contraind:
                answer += f"Противопоказания: {contraind[:200]}\n\n"
            if usage:
                answer += f"Применение: {usage[:300]}\n\n"
            if side_effects:
                answer += f"Побочные эффекты: {side_effects[:200]}\n"

            dataset.append({
                "text": question,
                "intent": "drug_info",
                "entity": name,
                "entity_id": drug_id,
                "answer": answer.strip(),
                "sources": [{"id": drug_id, "name": name}]
            })

    # 3. Вопросы об аналогах
    cur.execute("""
        SELECT d.id, d.trade_name, array_agg(s.name_ru) as substances
        FROM drugs d
        JOIN drug_substances ds ON d.id = ds.drug_id
        JOIN active_substances s ON ds.substance_id = s.id
        GROUP BY d.id, d.trade_name
        HAVING COUNT(ds.substance_id) > 0
        LIMIT 300;
    """)

    drugs_with_substances = cur.fetchall()

    for drug in tqdm(drugs_with_substances[:num_analogs], desc="Аналоги"):
        drug_id, name, substances = drug

        cur.execute("""
            SELECT d2.id, d2.trade_name, array_agg(s2.name_ru) as subs
            FROM drugs d2
            JOIN drug_substances ds2 ON d2.id = ds2.drug_id
            JOIN active_substances s2 ON ds2.substance_id = s2.id
            WHERE ds2.substance_id IN (
                SELECT substance_id FROM drug_substances WHERE drug_id = %s
            )
              AND d2.id != %s
            GROUP BY d2.id, d2.trade_name
            LIMIT 10;
        """, (drug_id, drug_id))

        analogs = cur.fetchall()

        if analogs:
            for template in random.sample(ANALOG_TEMPLATES, min(3, len(ANALOG_TEMPLATES))):
                question = template.format(drug_name=name)
                answer = f"💊 Аналоги препарата **{name}**"
                if substances:
                    answer += f" (активные вещества: {', '.join(substances[:3])})"
                answer += ":\n\n"
                for i, analog in enumerate(analogs[:10], 1):
                    answer += f"{i}. {analog[1]}"
                    if analog[2]:
                        answer += f" ({', '.join(analog[2][:3])})"
                    answer += "\n"

                dataset.append({
                    "text": question,
                    "intent": "analogs",
                    "entity": name,
                    "entity_id": drug_id,
                    "answer": answer.strip(),
                    "sources": [{"id": drug_id, "name": name}] +
                               [{"id": a[0], "name": a[1]} for a in analogs[:5]]
                })

    # 4. Аугментация данных
    augmented = []
    for item in dataset:
        augmented.append(item)

        if item['intent'] == 'symptom':
            variants = [
                item['text'].replace("Что делать", "Как лечить"),
                item['text'].replace("Что принимать", "Что пить"),
            ]
        elif item['intent'] == 'drug_info':
            variants = [
                item['text'].replace("Расскажи", "Расскажите"),
                item['text'].replace("Что такое", "Что знаете про"),
            ]
        elif item['intent'] == 'analogs':
            variants = [
                item['text'].replace("Какие аналоги", "Есть ли аналоги"),
                item['text'].replace("Чем заменить", "На что заменить"),
            ]
        else:
            variants = []

        for variant in variants[:2]:
            augmented.append({**item, "text": variant})

    dataset = augmented

    # 5. Сохранение — разбивка на train/val/test (80/10/10)
    random.shuffle(dataset)
    n = len(dataset)
    train_size = int(n * 0.8)
    val_size = int(n * 0.1)

    with open("../data/dataset_train.json", "w", encoding="utf-8") as f:
        json.dump(dataset[:train_size], f, ensure_ascii=False, indent=2)

    with open("../data/dataset_val.json", "w", encoding="utf-8") as f:
        json.dump(dataset[train_size:train_size + val_size], f, ensure_ascii=False, indent=2)

    with open("../data/dataset_test.json", "w", encoding="utf-8") as f:
        json.dump(dataset[train_size + val_size:], f, ensure_ascii=False, indent=2)

    cur.close()
    conn.close()

    return dataset


if __name__ == "__main__":
    generate_dataset()