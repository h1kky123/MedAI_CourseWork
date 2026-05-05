"""
Генерация датасета для обучения нейросети
Создаёт вопросы и ответы из данных в БД
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

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

# Шаблоны вопросов
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
    """Генерация датасета из БД"""
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    dataset = []
    
    # ============================================================
    # 1. СИМПТОМЫ — поиск препаратов по показаниям
    # ============================================================
    print("\n1. Генерация вопросов по симптомам...")
    
    # Извлекаем уникальные показания из БД
    cur.execute("""
        SELECT DISTINCT indications FROM drug_content 
        WHERE indications IS NOT NULL AND length(indications) > 20
        LIMIT 1000;
    """)
    
    indications = [r[0] for r in cur.fetchall()]
    
    # Словарь симптомов для поиска
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
        # Ищем препараты с такими показаниями
        # Для to_tsquery: разделяем слова на отдельные токены
        all_tokens = []
        for kw in keywords:
            tokens = kw.split()
            all_tokens.extend(tokens)
        query_str = " | ".join(all_tokens[:5])  # максимум 5 токенов
        
        cur.execute("""
            SELECT d.id, d.trade_name, d.manufacturer, dc.indications, dc.description
            FROM drugs d
            JOIN drug_content dc ON d.id = dc.drug_id
            WHERE dc.search_vector @@ to_tsquery('russian', %s)
            LIMIT 10;
        """, (query_str,))
        
        drugs_for_symptom = cur.fetchall()
        
        if drugs_for_symptom:
            # Создаём несколько вариантов вопросов
            for template in random.sample(SYMPTOM_TEMPLATES, min(5, len(SYMPTOM_TEMPLATES))):
                question = template.format(symptom=symptom)
                
                # Формируем ответ
                answer = f"При {symptom} могут помочь следующие препараты:\n\n"
                for i, drug in enumerate(drugs_for_symptom[:5], 1):
                    answer += f"{i}. {drug[1]}"  # trade_name
                    if drug[3]:  # indications
                        answer += f" — {drug[3][:150]}..."
                    answer += "\n"
                
                dataset.append({
                    "text": question,
                    "intent": "symptom",
                    "entity": symptom,
                    "answer": answer,
                    "sources": [{"id": d[0], "name": d[1]} for d in drugs_for_symptom[:5]]
                })
    
    print(f"  Создано вопросов по симптомам: {len([d for d in dataset if d['intent'] == 'symptom'])}")
    
    # ============================================================
    # 2. ПРЕПАРАТЫ — информация о конкретном лекарстве
    # ============================================================
    print("\n2. Генерация вопросов о препаратах...")
    
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
    
    drug_start = len(dataset)
    for drug in tqdm(drugs[:num_drugs], desc="Препараты"):
        drug_id, name, manufacturer, desc, indications, contraind, usage, side_effects = drug
        
        # Создаём 3-5 вопросов про каждый препарат
        num_questions = random.randint(3, 5)
        templates = random.sample(DRUG_TEMPLATES, num_questions)
        
        for template in templates:
            question = template.format(drug_name=name)
            
            # Формируем ответ из данных БД
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
    
    print(f"  Создано вопросов о препаратах: {len([d for d in dataset if d['intent'] == 'drug_info']) - drug_start}")
    
    # ============================================================
    # 3. АНАЛОГИ — поиск аналогов препарата
    # ============================================================
    print("\n3. Генерация вопросов об аналогах...")
    
    # Находим препараты с активными веществами
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
    
    analog_start = len(dataset)
    for drug in tqdm(drugs_with_substances[:num_analogs], desc="Аналоги"):
        drug_id, name, substances = drug
        
        # Ищем аналоги по веществам
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
            # Создаём вопросы про аналоги
            for template in random.sample(ANALOG_TEMPLATES, min(3, len(ANALOG_TEMPLATES))):
                question = template.format(drug_name=name)
                
                # Формируем ответ
                answer = f"💊 Аналоги препарата **{name}**"
                if substances:
                    subs_str = ", ".join(substances[:3])
                    answer += f" (активные вещества: {subs_str})"
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
    
    print(f"  Создано вопросов об аналогах: {len([d for d in dataset if d['intent'] == 'analogs']) - analog_start}")
    
    # ============================================================
    # 4. ДОБАВЛЯЕМ ВАРИАЦИИ (аугментация данных)
    # ============================================================
    print("\n4. Аугментация данных...")
    
    augmented = []
    for item in dataset:
        # Добавляем оригинал
        augmented.append(item)
        
        # Создаём варианты с разной формулировкой
        if item['intent'] == 'symptom':
            variants = [
                item['text'].replace("Что делать", "Как лечить"),
                item['text'].replace("Что принимать", "Что пить"),
                item['text'].replace("?", " please"),
            ]
            for variant in variants[:2]:
                augmented.append({
                    **item,
                    "text": variant
                })
        
        elif item['intent'] == 'drug_info':
            variants = [
                item['text'].replace("Расскажи", "Расскажите"),
                item['text'].replace("Что такое", "Что знаете про"),
            ]
            for variant in variants[:2]:
                augmented.append({
                    **item,
                    "text": variant
                })
        
        elif item['intent'] == 'analogs':
            variants = [
                item['text'].replace("Какие аналоги", "Есть ли аналоги"),
                item['text'].replace("Чем заменить", "На что заменить"),
            ]
            for variant in variants[:2]:
                augmented.append({
                    **item,
                    "text": variant
                })
    
    dataset = augmented
    print(f"  После аугментации: {len(dataset)} примеров")
    
    # ============================================================
    # 5. СОХРАНЕНИЕ
    # ============================================================
    print("\n5. Сохранение датасета...")
    
    # Разделяем на train/val/test
    random.shuffle(dataset)
    n = len(dataset)
    train_size = int(n * 0.8)
    val_size = int(n * 0.1)
    
    train_data = dataset[:train_size]
    val_data = dataset[train_size:train_size + val_size]
    test_data = dataset[train_size + val_size:]
    
    # Сохраняем
    with open("dataset_train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open("dataset_val.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    with open("dataset_test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    # Статистика
    print(f"\n{'='*60}")
    print(f"ДАТАСЕТ СОЗДАН!")
    print(f"{'='*60}")
    print(f"Train: {len(train_data)} примеров")
    print(f"Val:   {len(val_data)} примеров")
    print(f"Test:  {len(test_data)} примеров")
    print(f"Total: {len(dataset)} примеров")
    
    intents_count = {}
    for item in dataset:
        intents_count[item['intent']] = intents_count.get(item['intent'], 0) + 1
    
    print(f"\nРаспределение по классам:")
    for intent, count in sorted(intents_count.items()):
        print(f"  {intent}: {count}")
    
    cur.close()
    conn.close()
    
    return dataset


if __name__ == "__main__":
    print("="*60)
    print("ГЕНЕРАЦИЯ ДАТАСЕТА ДЛЯ НЕЙРОСЕТИ")
    print("="*60)
    
    dataset = generate_dataset()
    
    print(f"\n✓ Датасет сохранён в:")
    print(f"  - dataset_train.json")
    print(f"  - dataset_val.json")
    print(f"  - dataset_test.json")
