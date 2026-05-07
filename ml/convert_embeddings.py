import psycopg2, numpy as np
from tqdm import tqdm

c = psycopg2.connect(dbname='medicines_db',user='postgres',password='bkmzrjh1231',host='localhost',port='5433')
cur = c.cursor()

print("1. Удаление старой колонки FLOAT[]...")
cur.execute("ALTER TABLE drug_content DROP COLUMN IF EXISTS embedding_v")
cur.execute("ALTER TABLE drug_content ADD COLUMN embedding_v vector(384)")
c.commit()

print("2. Копирование данных...")
cur.execute("SELECT drug_id, embedding FROM drug_content WHERE embedding IS NOT NULL")
rows = cur.fetchall()
print(f"  Записей: {len(rows)}")

for drug_id, emb_float in tqdm(rows, desc="Конвертация"):
    emb_list = list(emb_float) if hasattr(emb_float, '__iter__') else emb_float
    cur.execute("UPDATE drug_content SET embedding_v = %s::vector(384) WHERE drug_id = %s", (emb_list, drug_id))
    if drug_id % 500 == 0:
        c.commit()
c.commit()

print("3. Замена колонки...")
cur.execute("ALTER TABLE drug_content DROP COLUMN embedding")
cur.execute("ALTER TABLE drug_content RENAME COLUMN embedding_v TO embedding")
c.commit()

print("4. Создание индекса...")
cur.execute("SELECT EXISTS(SELECT 1 FROM pg_indexes WHERE indexname='idx_drug_embedding')")
if not cur.fetchone()[0]:
    cur.execute("CREATE INDEX idx_drug_embedding ON drug_content USING ivfflat(embedding vector_cosine_ops) WITH(lists=100)")
    c.commit()
    print("Индекс создан")

cur.execute("SELECT COUNT(*) FROM drug_content WHERE embedding IS NOT NULL")
print(f"\nГотово: {cur.fetchone()[0]} записей с vector(384)")
cur.close(); c.close()
