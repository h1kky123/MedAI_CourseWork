"""
Загрузка нового датасета в БД с полной очисткой старых данных
"""
import json
import psycopg2
from tqdm import tqdm

DB_CONFIG = {
    "dbname": "medicines_db",
    "user": "postgres",
    "password": "bkmzrjh1231",
    "host": "localhost",
    "port": "5433"
}

JSON_FILE = "vidal_dataset_20260411_013745.json"

def main():
    print("="*60)
    print("ЗАГРУЗКА НОВОГО ДАТАСЕТА В БД")
    print("="*60)
    
    # 1. Очистка БД
    print("\n1. Очистка старых данных...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    cur.execute("DELETE FROM drug_content;")
    cur.execute("DELETE FROM drug_substances;")
    cur.execute("DELETE FROM drug_forms;")
    cur.execute("DELETE FROM drugs;")
    cur.execute("DELETE FROM active_substances;")
    # Сброс序列
    cur.execute("ALTER SEQUENCE drugs_id_seq RESTART WITH 1;")
    cur.execute("ALTER SEQUENCE active_substances_id_seq RESTART WITH 1;")
    cur.execute("ALTER SEQUENCE drug_forms_id_seq RESTART WITH 1;")
    
    conn.commit()
    print("  ✓ БД очищена")
    
    # 2. Загрузка JSON
    print(f"\n2. Загрузка {JSON_FILE}...")
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    print(f"  ✓ Загружено {len(dataset)} препаратов")
    
    # 3. Сохранение в БД
    print("\n3. Сохранение в БД...")
    drugs_saved = 0
    substances_saved = 0
    
    for drug in tqdm(dataset, desc="Сохранение"):
        try:
            cur.execute("""
                INSERT INTO drugs (trade_name, source_url, manufacturer, reg_number, prescription_status)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id;
            """, (
                drug.get('trade_name', ''),
                drug.get('source_url', ''),
                drug.get('manufacturer', ''),
                drug.get('reg_number', ''),
                drug.get('prescription_status', '')
            ))
            
            drug_id = cur.fetchone()[0]
            drugs_saved += 1
            
            # Активные вещества
            for substance in drug.get('active_substances', []):
                cur.execute("""
                    INSERT INTO active_substances (name_ru, name_en)
                    VALUES (%s, %s)
                    ON CONFLICT (name_ru) DO UPDATE SET name_en = EXCLUDED.name_en
                    RETURNING id;
                """, (substance.get('name_ru', ''), substance.get('name_en', '')))
                
                substance_id = cur.fetchone()[0]
                substances_saved += 1
                
                cur.execute("""
                    INSERT INTO drug_substances (drug_id, substance_id, dosage)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (drug_id, substance_id) DO NOTHING;
                """, (drug_id, substance_id, substance.get('dosage', '')))
            
            # Формы выпуска
            for dosage_info in drug.get('dosages', []):
                cur.execute("""
                    INSERT INTO drug_forms (drug_id, form_description, dosage, package_size)
                    VALUES (%s, %s, %s, %s);
                """, (
                    drug_id,
                    drug.get('trade_name', ''),
                    dosage_info.get('dosage', ''),
                    dosage_info.get('package_size', '')
                ))
            
            # Контент
            cur.execute("""
                INSERT INTO drug_content (
                    drug_id, description, indications, contraindications,
                    side_effects, precautions, usage_instructions,
                    interactions, special_instructions, storage_conditions
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """, (
                drug_id,
                drug.get('description', ''),
                drug.get('indications', ''),
                drug.get('contraindications', ''),
                drug.get('side_effects', ''),
                drug.get('precautions', ''),
                drug.get('usage_instructions', ''),
                drug.get('interactions', ''),
                drug.get('special_instructions', ''),
                drug.get('storage_conditions', '')
            ))
            
            # Коммит каждые 500 записей
            if drugs_saved % 500 == 0:
                conn.commit()
        
        except Exception as e:
            conn.rollback()
            continue
    
    conn.commit()
    
    # Обновление search_vector
    print("\n4. Обновление полнотекстового индекса...")
    cur.execute("""
        UPDATE drug_content
        SET search_vector = 
            setweight(to_tsvector('russian', coalesce(description, '')), 'A') ||
            setweight(to_tsvector('russian', coalesce(indications, '')), 'B') ||
            setweight(to_tsvector('russian', coalesce(contraindications, '')), 'C') ||
            setweight(to_tsvector('russian', coalesce(side_effects, '')), 'D');
    """)
    conn.commit()
    print("  ✓ Индекс обновлён")
    
    # 5. Статистика
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТ:")
    print("="*60)
    
    cur.execute("SELECT COUNT(*) FROM drugs;")
    print(f"  Препаратов: {cur.fetchone()[0]}")
    
    cur.execute("SELECT COUNT(*) FROM active_substances;")
    print(f"  Активных веществ: {cur.fetchone()[0]}")
    
    cur.execute("SELECT COUNT(*) FROM drug_substances;")
    print(f"  Связей препарат-вещество: {cur.fetchone()[0]}")
    
    cur.execute("SELECT COUNT(*) FROM drug_forms;")
    print(f"  Форм выпуска: {cur.fetchone()[0]}")
    
    cur.execute("SELECT COUNT(*) FROM drug_content;")
    print(f"  Записей контента: {cur.fetchone()[0]}")
    
    # Топ веществ
    cur.execute("""
        SELECT s.name_ru, COUNT(*) as cnt
        FROM active_substances s
        JOIN drug_substances ds ON s.id = ds.substance_id
        GROUP BY s.id, s.name_ru
        ORDER BY cnt DESC
        LIMIT 10;
    """)
    print(f"\n  Топ-10 веществ:")
    for name, count in cur.fetchall():
        print(f"    {name}: {count}")
    
    cur.close()
    conn.close()
    
    print(f"\n{'='*60}")
    print("✓ ЗАГРУЗКА ЗАВЕРШЕНА!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
