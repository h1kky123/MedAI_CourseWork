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
        self._setup_pgvector()
        self._ensure_model()

    def _setup_pgvector(self):
        # Проверка и настройка расширения pgvector
        try:
            with self.connection.cursor() as c:
                c.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname='vector')")
                if c.fetchone()[0]:
                    self.use_pgvector = True
                    c.execute("SELECT EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name='drug_content' AND column_name='embedding')")
                    if not c.fetchone()[0]:
                        c.execute(f"ALTER TABLE drug_content ADD COLUMN embedding vector({self.embedding_dim})")
                        self.connection.commit()
                    c.execute("SELECT EXISTS(SELECT 1 FROM pg_indexes WHERE indexname='idx_drug_embedding')")
                    if not c.fetchone()[0]:
                        c.execute("CREATE INDEX idx_drug_embedding ON drug_content USING ivfflat(embedding vector_cosine_ops) WITH(lists=100)")
                        self.connection.commit()
        except Exception as e:
            self.use_pgvector = False

    def close(self):
        if self.connection:
            self.connection.close()

    def _ensure_model(self):
        if self.model is None:
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def generate_embedding(self, text):
        self._ensure_model()
        return self.model.encode(text).tolist()

    def search_drugs_fulltext(self, query, top_k=10):
        # Полнотекстовый поиск по препаратам
        try:
            with self.connection.cursor() as c:
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

                return [{
                    'id': r[0], 'trade_name': r[1], 'source_url': r[2],
                    'manufacturer': r[3], 'description': r[4],
                    'indications': r[5], 'contraindications': r[6], 'rank': float(r[7])
                } for r in c.fetchall()]
        except Exception:
            return []

    def search_drugs_semantic(self, query, top_k=10):
        # Семантический поиск по векторным эмбеддингам
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
                        'id': r[0], 'trade_name': r[1], 'source_url': r[2],
                        'manufacturer': r[3], 'description': r[4],
                        'indications': r[5], 'contraindications': r[6], 'similarity': float(r[7])
                    } for r in c.fetchall()]
                else:
                    # Fallback: косинусная схожесть вручную без pgvector
                    c.execute("""
                        SELECT d.id, d.trade_name, d.source_url, d.manufacturer,
                               dc.description, dc.indications, dc.contraindications, dc.embedding
                        FROM drugs d
                        JOIN drug_content dc ON d.id=dc.drug_id
                        WHERE dc.embedding IS NOT NULL
                    """)
                    res = []
                    for r in c.fetchall():
                        db_emb = np.array(r[7])
                        query_emb = np.array(emb)
                        norm_db = np.linalg.norm(db_emb)
                        norm_query = np.linalg.norm(query_emb)
                        if norm_db > 0 and norm_query > 0:
                            sim = float(np.dot(query_emb, db_emb) / (norm_db * norm_query))
                            res.append({
                                'id': r[0], 'trade_name': r[1], 'source_url': r[2],
                                'manufacturer': r[3], 'description': r[4],
                                'indications': r[5], 'contraindications': r[6], 'similarity': sim
                            })
                    res.sort(key=lambda x: x['similarity'], reverse=True)
                    return res[:top_k]
        except Exception:
            return []

    def get_drug_details(self, drug_id):
        # Получение полной карточки препарата по ID
        try:
            with self.connection.cursor() as c:
                c.execute("""
                    SELECT id, trade_name, source_url, manufacturer, reg_number, prescription_status
                    FROM drugs WHERE id=%s
                """, (drug_id,))
                r = c.fetchone()
                if not r:
                    return None

                res = {
                    'id': r[0], 'trade_name': r[1], 'source_url': r[2],
                    'manufacturer': r[3], 'reg_number': r[4], 'prescription_status': r[5]
                }

                c.execute("""
                    SELECT description, indications, contraindications, side_effects,
                           precautions, usage_instructions, interactions,
                           special_instructions, storage_conditions
                    FROM drug_content WHERE drug_id=%s
                """, (drug_id,))
                dc = c.fetchone()
                if dc:
                    res.update({
                        'description': dc[0], 'indications': dc[1],
                        'contraindications': dc[2], 'side_effects': dc[3],
                        'precautions': dc[4], 'usage_instructions': dc[5],
                        'interactions': dc[6], 'special_instructions': dc[7],
                        'storage_conditions': dc[8]
                    })

                c.execute("""
                    SELECT s.name_ru, s.name_en
                    FROM drug_substances ds
                    JOIN active_substances s ON ds.substance_id=s.id
                    WHERE ds.drug_id=%s
                """, (drug_id,))
                res['substances'] = [{'name_ru': row[0], 'name_en': row[1]} for row in c.fetchall()]

                return res
        except Exception:
            return None

    def get_analogs_by_substance(self, drug_id, top_k=10):
        # Поиск структурных аналогов по активным веществам
        try:
            with self.connection.cursor() as c:
                c.execute("SELECT substance_id FROM drug_substances WHERE drug_id = %s", (drug_id,))
                substance_ids = [r[0] for r in c.fetchall()]
                if not substance_ids:
                    return []

                c.execute("""
                    SELECT d.id, d.trade_name, d.source_url, d.manufacturer,
                           array_agg(s.name_ru) as substances,
                           COUNT(DISTINCT ds.substance_id) as common_count
                    FROM drugs d
                    JOIN drug_substances ds ON d.id = ds.drug_id
                    JOIN active_substances s ON ds.substance_id = s.id
                    WHERE ds.substance_id = ANY(%s) AND d.id != %s
                    GROUP BY d.id, d.trade_name, d.source_url, d.manufacturer
                    ORDER BY common_count DESC, d.trade_name
                    LIMIT %s;
                """, (substance_ids, drug_id, top_k))

                return [{
                    'id': r[0], 'trade_name': r[1], 'source_url': r[2],
                    'manufacturer': r[3], 'substances': r[4], 'common_count': r[5]
                } for r in c.fetchall()]
        except Exception:
            return []

    def get_analogs_by_atc(self, drug_id, top_k=10):
        # Поиск терапевтических аналогов по схожести показаний
        try:
            with self.connection.cursor() as c:
                c.execute("SELECT dc.indications FROM drug_content dc WHERE dc.drug_id = %s", (drug_id,))
                result = c.fetchone()
                if not result or not result[0]:
                    return []

                indications = result[0][:200]
                c.execute("""
                    SELECT d.id, d.trade_name, d.source_url, d.manufacturer,
                           ts_rank(dc.search_vector, plainto_tsquery('russian', %s)) as rank
                    FROM drugs d
                    JOIN drug_content dc ON d.id = dc.drug_id
                    WHERE dc.search_vector @@ plainto_tsquery('russian', %s) AND d.id != %s
                    ORDER BY rank DESC
                    LIMIT %s;
                """, (indications, indications, drug_id, top_k))

                return [{
                    'id': r[0], 'trade_name': r[1], 'source_url': r[2],
                    'manufacturer': r[3], 'common_atc_count': int(r[4] * 10)
                } for r in c.fetchall()]
        except Exception:
            return []

    def get_statistics(self):
        try:
            with self.connection.cursor() as c:
                c.execute("SELECT COUNT(*) FROM drugs")
                drugs = c.fetchone()[0]
                c.execute("SELECT COUNT(*) FROM active_substances")
                subs = c.fetchone()[0]
                return {'total_drugs': drugs, 'total_substances': subs}
        except Exception:
            return {}