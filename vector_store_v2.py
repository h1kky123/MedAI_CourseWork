import os
import json
import numpy as np
from typing import List, Dict, Optional
import psycopg2
from sentence_transformers import SentenceTransformer

class PostgreSQLVectorStoreV2:
    def __init__(self, db_config=None):
        self.db_config = db_config or {
            "dbname": "medicines_db",
            "user": "postgres",
            "password": "bkmzrjh1231",
            "host": "localhost",
            "port": "5433"
        }
        self.model = None
        self.connection = None
        self.use_pgvector = False
        self.embedding_dim = 384

    def connect(self):
        self.connection = psycopg2.connect(**self.db_config)
        print("OK PostgreSQL подключён")
        self._setup_pgvector()
        self._ensure_model()

    def _setup_pgvector(self):
        try:
            with self.connection.cursor() as c:
                # Проверка расширения vector
                c.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname='vector')")
                if c.fetchone()[0]:
                    self.use_pgvector = True
                    print("OK pgvector найден")

                    # Проверка колонки embedding
                    c.execute("SELECT EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name='drug_content' AND column_name='embedding')")
                    if not c.fetchone()[0]:
                        c.execute(f"ALTER TABLE drug_content ADD COLUMN embedding vector({self.embedding_dim})")
                        self.connection.commit()
                        print("  OK колонка embedding добавлена")

                    # Проверка индекса
                    c.execute("SELECT EXISTS(SELECT 1 FROM pg_indexes WHERE indexname='idx_drug_embedding')")
                    if not c.fetchone()[0]:
                        c.execute(f"CREATE INDEX idx_drug_embedding ON drug_content USING ivfflat(embedding vector_cosine_ops) WITH(lists=100)")
                        self.connection.commit()
                        print("  OK индекс создан")
                else:
                    print("WARNING pgvector НЕ найден")
        except Exception as e:
            print(f"WARNING pgvector setup error: {e}")
            self.use_pgvector = False

    def close(self):
        if self.connection:
            self.connection.close()
            print("OK Соединение с БД закрыто")

    def _ensure_model(self):
        if self.model is None:
            print("Загрузка модели эмбеддингов...")
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def generate_embedding(self, text):
        self._ensure_model()
        return self.model.encode(text).tolist()

    def search_drugs_fulltext(self, query, top_k=10):
        """Полнотекстовый поиск с защитой от синтаксических ошибок SQL"""
        try:
            with self.connection.cursor() as c:
                # Используем plainto_tsquery вместо ручной замены пробелов.
                # Это безопаснее и автоматически обрабатывает морфологию.
                c.execute("""
                    SELECT d.id, d.trade_name, d.source_url, d.manufacturer,
                           dc.description, dc.indications, dc.contraindications,
                           ts_rank(dc.search_vector, plainto_tsquery('russian', %s)) as rank
                    FROM drugs d 
                    JOIN drug_content dc ON d.id=dc.drug_id
                    WHERE dc.search_vector @@ plainto_tsquery('russian', %s)
                    ORDER BY ts_rank(dc.search_vector, plainto_tsquery('russian', %s)) DESC 
                    LIMIT %s;
                """, (query, query, query, top_k))

                results = []
                for r in c.fetchall():
                    results.append({
                        'id': r[0],
                        'trade_name': r[1],
                        'source_url': r[2],
                        'manufacturer': r[3],
                        'description': r[4],
                        'indications': r[5],
                        'contraindications': r[6],
                        'rank': float(r[7])
                    })
                return results
        except Exception as e:
            print(f"Ошибка FTS поиска: {e}")
            return []

    def search_drugs_semantic(self, query, top_k=10):
        """Векторный поиск по смыслу"""
        try:
            emb = self.generate_embedding(query)
            with self.connection.cursor() as c:
                if self.use_pgvector:
                    c.execute("""
                        SELECT d.id, d.trade_name, d.source_url, d.manufacturer,
                               dc.description, dc.indications, dc.contraindications,
                               1-(dc.embedding <=> %s::vector) as sim
                        FROM drugs d 
                        JOIN drug_content dc ON d.id=dc.drug_id
                        WHERE dc.embedding IS NOT NULL
                        ORDER BY dc.embedding <=> %s::vector 
                        LIMIT %s;
                    """, (emb, emb, top_k))

                    return [{
                        'id': r[0],
                        'trade_name': r[1],
                        'source_url': r[2],
                        'manufacturer': r[3],
                        'description': r[4],
                        'indications': r[5],
                        'contraindications': r[6],
                        'similarity': float(r[7])
                    } for r in c.fetchall()]
                else:
                    # Fallback для CPU без pgvector (медленно, но работает)
                    c.execute("""
                        SELECT d.id, d.trade_name, d.source_url, d.manufacturer,
                               dc.description, dc.indications, dc.contraindications, dc.embedding 
                        FROM drugs d 
                        JOIN drug_content dc ON d.id=dc.drug_id 
                        WHERE dc.embedding IS NOT NULL
                    """)
                    res = []
                    for r in c.fetchall():
                        # Вычисляем косинусную схожесть вручную
                        db_emb = np.array(r[7])
                        query_emb = np.array(emb)
                        norm_db = np.linalg.norm(db_emb)
                        norm_query = np.linalg.norm(query_emb)

                        if norm_db > 0 and norm_query > 0:
                            sim = float(np.dot(query_emb, db_emb) / (norm_db * norm_query))
                            res.append({
                                'id': r[0],
                                'trade_name': r[1],
                                'source_url': r[2],
                                'manufacturer': r[3],
                                'description': r[4],
                                'indications': r[5],
                                'contraindications': r[6],
                                'similarity': sim
                            })

                    res.sort(key=lambda x: x['similarity'], reverse=True)
                    return res[:top_k]
        except Exception as e:
            print(f"Ошибка семантического поиска: {e}")
            return []

    def get_drug_details(self, drug_id):
        """Получение полной карточки препарата"""
        try:
            with self.connection.cursor() as c:
                # Основная информация
                c.execute("""
                    SELECT id, trade_name, source_url, manufacturer, reg_number, prescription_status 
                    FROM drugs WHERE id=%s
                """, (drug_id,))
                r = c.fetchone()
                if not r:
                    return None

                res = {
                    'id': r[0],
                    'trade_name': r[1],
                    'source_url': r[2],
                    'manufacturer': r[3],
                    'reg_number': r[4],
                    'prescription_status': r[5]
                }

                # Контент
                c.execute("""
                    SELECT description, indications, contraindications, side_effects,
                           precautions, usage_instructions, interactions, 
                           special_instructions, storage_conditions 
                    FROM drug_content WHERE drug_id=%s
                """, (drug_id,))
                dc = c.fetchone()
                if dc:
                    res.update({
                        'description': dc[0],
                        'indications': dc[1],
                        'contraindications': dc[2],
                        'side_effects': dc[3],
                        'precautions': dc[4],
                        'usage_instructions': dc[5],
                        'interactions': dc[6],
                        'special_instructions': dc[7],
                        'storage_conditions': dc[8]
                    })

                # Активные вещества
                c.execute("""
                    SELECT s.name_ru, s.name_en 
                    FROM drug_substances ds 
                    JOIN active_substances s ON ds.substance_id=s.id 
                    WHERE ds.drug_id=%s
                """, (drug_id,))
                res['substances'] = [{'name_ru': row[0], 'name_en': row[1]} for row in c.fetchall()]

                return res
        except Exception as e:
            print(f"Ошибка получения деталей препарата {drug_id}: {e}")
            return None

    def get_analogs_by_substance(self, drug_id, top_k=10):
        """Поиск структурных аналогов по активным веществам"""
        try:
            with self.connection.cursor() as c:
                # Находим ID активных веществ текущего препарата
                c.execute("""
                    SELECT substance_id FROM drug_substances WHERE drug_id = %s
                """, (drug_id,))
                substance_ids = [r[0] for r in c.fetchall()]

                if not substance_ids:
                    return []

                # Ищем другие препараты, содержащие эти же вещества
                c.execute("""
                    SELECT d.id, d.trade_name, d.source_url, d.manufacturer,
                           array_agg(s.name_ru) as substances,
                           COUNT(DISTINCT ds.substance_id) as common_count
                    FROM drugs d
                    JOIN drug_substances ds ON d.id = ds.drug_id
                    JOIN active_substances s ON ds.substance_id = s.id
                    WHERE ds.substance_id = ANY(%s)
                      AND d.id != %s
                    GROUP BY d.id, d.trade_name, d.source_url, d.manufacturer
                    ORDER BY common_count DESC, d.trade_name
                    LIMIT %s;
                """, (substance_ids, drug_id, top_k))

                return [{
                    'id': r[0],
                    'trade_name': r[1],
                    'source_url': r[2],
                    'manufacturer': r[3],
                    'substances': r[4],
                    'common_count': r[5]
                } for r in c.fetchall()]
        except Exception as e:
            print(f"Ошибка поиска структурных аналогов: {e}")
            return []

    def get_analogs_by_atc(self, drug_id, top_k=10):
        """Поиск терапевтических аналогов (по показаниям/АТХ)"""
        try:
            with self.connection.cursor() as c:
                # Берем показания исходного препарата
                c.execute("""
                    SELECT dc.indications FROM drug_content dc WHERE dc.drug_id = %s
                """, (drug_id,))
                result = c.fetchone()

                if not result or not result[0]:
                    return []

                indications = result[0][:200] # Ограничиваем длину для запроса
                # plainto_tsquery безопаснее, чем ручная сборка строки
                c.execute("""
                    SELECT d.id, d.trade_name, d.source_url, d.manufacturer,
                           ts_rank(dc.search_vector, plainto_tsquery('russian', %s)) as rank
                    FROM drugs d
                    JOIN drug_content dc ON d.id = dc.drug_id
                    WHERE dc.search_vector @@ plainto_tsquery('russian', %s)
                      AND d.id != %s
                    ORDER BY rank DESC
                    LIMIT %s;
                """, (indications, indications, drug_id, top_k))

                return [{
                    'id': r[0],
                    'trade_name': r[1],
                    'source_url': r[2],
                    'manufacturer': r[3],
                    'common_atc_count': int(r[4] * 10) # Условный вес схожести
                } for r in c.fetchall()]
        except Exception as e:
            print(f"Ошибка поиска терапевтических аналогов: {e}")
            return []

    def get_statistics(self):
        try:
            with self.connection.cursor() as c:
                c.execute("SELECT COUNT(*) FROM drugs")
                drugs = c.fetchone()[0]
                c.execute("SELECT COUNT(*) FROM active_substances")
                subs = c.fetchone()[0]
                return {'total_drugs': drugs, 'total_substances': subs}
        except:
            return {}