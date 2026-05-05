"""
Генерация векторных эмбеддингов для всех препаратов в БД
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import psycopg2
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

DB_CONFIG = {
    "dbname": "medicines_db",
    "user": "postgres",
    "password": "bkmzrjh1231",
    "host": "localhost",
    "port": "5433"
}

BATCH_SIZE = 100

def main():
    print("="*60)
    print("ГЕНЕРАЦИЯ ЭМБЕДДИНГОВ")
    print("="*60)
    
    # 1. Добавляем колонку embedding
    print("\n1. Подготовка таблицы...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Проверяем есть ли колонка
    cur.execute("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns 
            WHERE table_name = 'drug_content' AND column_name = 'embedding'
        );
    """)
    has_column = cur.fetchone()[0]
    
    if not has_column:
        print("  Добавление колонки embedding...")
        cur.execute("ALTER TABLE drug_content ADD COLUMN embedding FLOAT[];")
        conn.commit()
        print("  OK Колонка добавлена")
    else:
        print("  Очистка старых эмбеддингов...")
        cur.execute("UPDATE drug_content SET embedding = NULL;")
        conn.commit()
        print("  OK Очищено")
    
    # 2. Загружаем модель
    print("\n2. Загрузка модели...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("  OK Модель загружена")
    
    # 3. Получаем все записи без эмбеддингов
    print("\n3. Получение данных...")
    cur.execute("""
        SELECT dc.drug_id, d.trade_name, 
               COALESCE(dc.description, '') || ' ' || 
               COALESCE(dc.indications, '') || ' ' ||
               COALESCE(dc.contraindications, '') as content
        FROM drug_content dc
        JOIN drugs d ON dc.drug_id = d.id
        WHERE dc.embedding IS NULL;
    """)
    
    records = cur.fetchall()
    print(f"  Найдено {len(records)} записей")
    
    if not records:
        print("  Все эмбеддинги уже сгенерированы!")
        cur.close()
        conn.close()
        return
    
    # 4. Генерация эмбеддингов батчами
    print(f"\n4. Генерация эмбеддингов (батчи по {BATCH_SIZE})...")
    
    total_processed = 0
    for i in tqdm(range(0, len(records), BATCH_SIZE), desc="Генерация"):
        batch = records[i:i + BATCH_SIZE]
        texts = [r[2][:2000] for r in batch]  # Ограничиваем длину текста
        
        try:
            embeddings = model.encode(texts, show_progress_bar=False)
            
            # Сохраняем в БД
            for idx, (drug_id, _, _) in enumerate(batch):
                embedding_list = embeddings[idx].tolist()
                cur.execute("""
                    UPDATE drug_content
                    SET embedding = %s
                    WHERE drug_id = %s;
                """, (embedding_list, drug_id))
            
            total_processed += len(batch)
            
            # Коммит каждые N батчей
            if (i // BATCH_SIZE + 1) % 10 == 0:
                conn.commit()
                tqdm.write(f"  Сохранено: {total_processed}/{len(records)}")
        
        except Exception as e:
            tqdm.write(f"  Ошибка в батче {i}: {e}")
            continue
    
    conn.commit()
    
    # 5. Статистика
    print("\n" + "="*60)
    cur.execute("SELECT COUNT(*) FROM drug_content WHERE embedding IS NOT NULL;")
    with_emb = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM drug_content;")
    total = cur.fetchone()[0]
    
    print(f"РЕЗУЛЬТАТ:")
    print(f"  С эмбеддингами: {with_emb}/{total}")
    print(f"  Без эмбеддингов: {total - with_emb}")
    
    cur.close()
    conn.close()
    
    print(f"\n{'='*60}")
    print("  OK Генерация завершена")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
