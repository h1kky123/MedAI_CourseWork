"""
FastAPI REST API для RAG-системы лекарственных препаратов
Версия 5.0 Final: Текстовый выбор из списка, умный поиск и генерация инструкций
"""
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import re
import torch
from typing import List, Optional, Dict
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pathlib

# Transformers & PEFT for LLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Local Modules
from vector_store_v2 import PostgreSQLVectorStoreV2
from drug_analogs_v2 import DrugAnalogsFinderV2
from intent_predictor import IntentPredictor


# ============================================================================
# МОДЕЛИ ДАННЫХ
# ============================================================================

class DrugSearchResult(BaseModel):
    id: int
    trade_name: str
    source_url: str
    manufacturer: Optional[str] = None
    description: Optional[str] = None
    indications: Optional[str] = None


class ChatRequest(BaseModel):
    query: str
    top_k: int = 10  # Увеличим топк, чтобы находить все варианты бренда


class ChatResponse(BaseModel):
    answer: str
    sources: List[DrugSearchResult] = []
    type: str  # 'selection', 'drug_info', 'analogs', 'symptom', 'general'


# ============================================================================
# ГЛОБАЛЬНЫЕ ОБЪЕКТЫ
# ============================================================================

vector_store: Optional[PostgreSQLVectorStoreV2] = None
analog_finder: Optional[DrugAnalogsFinderV2] = None
intent_classifier: Optional[IntentPredictor] = None
tokenizer = None
model = None  # Qwen + LoRA
model_device = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = pathlib.Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
LORA_PATH = BASE_DIR / "qwen_medical_lora"


# ============================================================================
# LIFESPAN (Инициализация)
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store, analog_finder, intent_classifier, tokenizer, model

    print("🚀 Инициализация системы v5.0...")

    # 1. БД
    try:
        vector_store = PostgreSQLVectorStoreV2()
        vector_store.connect()
        analog_finder = DrugAnalogsFinderV2()
        analog_finder.connect()
        print("✅ БД подключена")
    except Exception as e:
        print(f"❌ Ошибка БД: {e}")
        raise e

    # 2. Классификатор
    try:
        print("🧠 Загрузка Intent Classifier...")
        intent_classifier = IntentPredictor()
        print("✅ Intent Classifier готов")
    except Exception as e:
        print(f"⚠️ Warning Intent Classifier: {e}")
        intent_classifier = None

    # 3. LLM Qwen + LoRA
    try:
        print(f"💊 Загрузка Qwen2.5-1.5B + LoRA ({model_device})...")
        base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if model_device == "cuda" else torch.float32,
            device_map="auto" if model_device == "cuda" else None
        )

        if LORA_PATH.exists():
            model = PeftModel.from_pretrained(base_model, str(LORA_PATH))
            print("✅ LoRA адаптеры применены")
        else:
            model = base_model
            print("⚠️ LoRA не найдена, используется базовая модель")

        if model_device == "cpu":
            model.to(model_device)
        model.eval()
        print("✅ LLM готова")

    except Exception as e:
        print(f"❌ Ошибка LLM: {e}")
        raise e

    yield

    if vector_store: vector_store.close()
    if analog_finder: analog_finder.close()


# ============================================================================
# APP
# ============================================================================

app = FastAPI(title="MedAI RAG", version="5.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "API works. Check /docs"}


def generate_response_with_qwen(prompt_text: str, max_tokens: int = 500) -> str:
    """Генерация ответа через Fine-Tuned Qwen"""
    if not model or not tokenizer:
        return "Ошибка модели."

    messages = [{"role": "user", "content": prompt_text}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model_device)

    with torch.no_grad():
        generated = model.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = [out[len(inp):] for inp, out in zip(model_inputs.input_ids, generated)]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """RAG чат с логикой выбора из списка и точным поиском"""
    if not vector_store:
        raise HTTPException(status_code=503, detail="БД не подключена")

    try:
        query = request.query
        top_k = request.top_k
        print(f"\n=== ЗАПРОС: {query} ===")

        # ============================================================
        # 1. КЛАССИФИКАЦИЯ НАМЕРЕНИЯ
        # ============================================================

        intent = "general"
        entity = None

        if intent_classifier:
            intent_result = intent_classifier.predict(query)
            intent = intent_result['intent']
            confidence = intent_result.get('confidence', 0)
            entity = intent_result['entity']
            print(f"🧠 Intent: {intent}, Entity: {entity}")
        else:
            # Fallback логика если классификатор не загружен
            query_lower = query.lower()
            if any(w in query_lower for w in ["аналог", "заменит", "похож"]):
                intent = "analogs"
            elif any(w in query_lower for w in ["инструкц", "как примен", "дозировк"]):
                intent = "drug_info"
            elif any(w in query_lower for w in ["что дел", "что прин", "болит", "боль"]):
                intent = "symptom"
            else:
                intent = "general"
            entity = None

        # Очищаем сущность от мусора и стоп-слов
        if entity:
            stop_words_pattern = r'\b(инструкция|для|покажи|расскажи|про|о|что|такое|как)\b'
            entity_clean = re.sub(stop_words_pattern, '', entity, flags=re.IGNORECASE).strip()
            entity_clean = re.sub(r'[?!,;.)]', '', entity_clean).strip()
        else:
            entity_clean = query.strip()

        # ============================================================
        # 2. ПОИСК ПРЕПАРАТА (УЛУЧШЕННЫЙ С МЯГКИМ СРАВНЕНИЕМ)
        # ============================================================

        drug_id = None
        drug_name = None
        drug_details = None
        all_variants = []  # Список всех найденных вариантов бренда

        if entity and intent in ["analogs", "drug_info"]:
            words = [w for w in entity_clean.split() if len(w) > 1]

            with vector_store.connection.cursor() as c:
                # Ищем ВСЕ препараты, содержащие слова из запроса
                if words:
                    query_parts = []
                    params = []
                    for word in words:
                        query_parts.append("trade_name ILIKE %s")
                        params.append(f'%{word}%')

                    sql = f"SELECT id, trade_name FROM drugs WHERE {' AND '.join(query_parts)} LIMIT 20;"
                    c.execute(sql, params)
                    results = c.fetchall()
                    all_variants = [{'id': r[0], 'name': r[1]} for r in results]

                # Если по словам не нашли, ищем по частичному совпадению очищенного запроса
                if not all_variants and entity_clean:
                    c.execute("SELECT id, trade_name FROM drugs WHERE trade_name ILIKE %s LIMIT 20;",
                              (f'%{entity_clean}%',))
                    results = c.fetchall()
                    all_variants = [{'id': r[0], 'name': r[1]} for r in results]

                # ЛОГИКА ОБРАБОТКИ РЕЗУЛЬТАТОВ
                if len(all_variants) > 1:
                    best_match = None
                    best_score = -1

                    clean_req_normalized = re.sub(r'\s+', '', entity_clean.lower())
                    req_words = set(entity_clean.lower().split())

                    for v in all_variants:
                        # Очищаем название из БД: убираем ®, латинскую часть в скобках, лишние слова
                        db_name_clean = re.sub(r'\(.*?\)', '', v['name'])  # убираем всё в скобках
                        db_name_clean = re.sub(r'®|инструкция|по|применению', '', db_name_clean, flags=re.IGNORECASE)
                        db_name_clean = re.sub(r'\s+', ' ', db_name_clean).strip()

                        db_name_normalized = re.sub(r'\s+', '', db_name_clean.lower())
                        db_words = set(db_name_clean.lower().split())
                        score = 0

                        if clean_req_normalized == db_name_normalized:
                            score = 1000
                        elif db_name_normalized in clean_req_normalized:
                            score = 500
                        elif req_words.issubset(db_words):
                            score = 200 - len(db_words)
                        else:
                            common_words = req_words.intersection(db_words)
                            if common_words:
                                score = len(common_words) * 10 - len(db_words)

                        print(f"  Кандидат: {v['name']}, score={score}")

                        if score > best_score:
                            best_score = score
                            best_match = v

                    MIN_SCORE = 10
                    if best_match and best_score >= MIN_SCORE:
                        target_variant = best_match
                        print(f"✓ Лучший кандидат (score={best_score}): {target_variant['name']}")
                    else:
                        selection_text = f"Я нашел несколько препаратов, содержащих \"{entity_clean}\". Пожалуйста, уточните:\n\n"
                        for i, d in enumerate(all_variants[:10], 1):
                            selection_text += f"{i}. {d['name']}\n"
                        selection_text += "\nНапример, напишите: *Инструкция для Нурофен Экспресс*"

                        sources_obj = [DrugSearchResult(id=d['id'], trade_name=d['name'], source_url='') for d in
                                       all_variants[:10]]
                        return ChatResponse(answer=selection_text, sources=sources_obj, type="selection")

                elif len(all_variants) == 1:
                    target_variant = all_variants[0]
                else:
                    target_variant = None

                if target_variant:
                    drug_id = target_variant['id']
                    drug_name = target_variant['name']
                    print(f"✓ Выбран препарат ID={drug_id}: {drug_name}")
                    drug_details = vector_store.get_drug_details(drug_id)

        # ============================================================
        # 3. ФОРМИРОВАНИЕ ОТВЕТА
        # ============================================================

        # --- СЛУЧАЙ А: АНАЛОГИ ---
        if intent == "analogs" and drug_id and drug_details and analog_finder:
            structural = analog_finder.find_structural_analogs(drug_id, top_k=10)
            therapeutic = analog_finder.find_therapeutic_analogs(drug_id, top_k=5)

            answer = f"**Аналоги для препарата {drug_details['trade_name']}:**\n\n"

            if structural:
                answer += "**Структурные аналоги** (те же вещества):\n"
                for i, a in enumerate(structural[:10], 1):
                    subs = a.get('substances', [])
                    sub_str = f" ({', '.join(subs[:3])})" if subs else ""
                    answer += f"{i}. **{a['trade_name']}**{sub_str}\n"

            if therapeutic:
                answer += "\n**Терапевтические аналоги** (похожее действие):\n"
                for i, a in enumerate(therapeutic[:5], 1):
                    answer += f"{i}. {a['trade_name']}\n"

            if not structural and not therapeutic:
                answer += "Аналоги не найдены."

            sources = [
                DrugSearchResult(id=drug_id, trade_name=drug_name, source_url=drug_details.get('source_url', ''))]
            return ChatResponse(answer=answer, sources=sources, type="analogs")

        # --- СЛУЧАЙ Б: ИНСТРУКЦИЯ / ИНФО ---
        elif intent == "drug_info" and drug_id and drug_details:
            answer = f"**Информация о препарате {drug_details['trade_name']}:**\n\n"

            if drug_details.get('substances'):
                subs = ", ".join([s['name_ru'] for s in drug_details['substances']])
                answer += f"**Активные вещества:** {subs}\n\n"

            if drug_details.get('indications'):
                answer += f"**Показания:** {drug_details['indications'][:400]}\n\n"

            if drug_details.get('contraindications'):
                answer += f"**Противопоказания:** {drug_details['contraindications'][:300]}\n\n"

            if drug_details.get('usage_instructions'):
                answer += f"**Применение:** {drug_details['usage_instructions'][:400]}\n\n"

            if drug_details.get('side_effects'):
                answer += f"**Побочные эффекты:** {drug_details['side_effects'][:300]}\n\n"

            # Добавляем футер с другими вариантами, если они были найдены
            other_variants = [v for v in all_variants if v['id'] != drug_id]
            if other_variants:
                answer += "\n**Другие формы этого препарата:**\n"
                for v in other_variants[:5]:
                    # Очищаем название: убираем скобки с латиницей и "инструкция по применению"
                    clean_name = re.sub(r'\(.*?\)', '', v['name'])
                    clean_name = re.sub(r'инструкция по применению', '', clean_name, flags=re.IGNORECASE)
                    clean_name = re.sub(r'®', ' ', clean_name)
                    clean_name = re.sub(r'\s+', ' ', clean_name).strip()
                    answer += f"• {clean_name}\n"

            sources = [
                DrugSearchResult(id=drug_id, trade_name=drug_name, source_url=drug_details.get('source_url', ''))]
            print(f"=== ANSWER ===\n{answer}\n=== END ===")
            return ChatResponse(answer=answer, sources=sources, type="drug_info")

        # --- СЛУЧАЙ В: СИМПТОМЫ / ОБЩИЙ ПОИСК ---
        else:
            print("🔍 Hybrid Search (Symptom/General)...")
            results_fts = vector_store.search_drugs_fulltext(query, top_k=top_k)
            results_sem = vector_store.search_drugs_semantic(query, top_k=top_k)

            seen_ids = set()
            combined = []
            for r in results_fts + results_sem:
                if r['id'] not in seen_ids:
                    seen_ids.add(r['id'])
                    combined.append(r)
                    if len(combined) >= 5: break

            if not combined:
                return ChatResponse(answer="К сожалению, ничего не найдено.", sources=[], type="general")

            answer = "⚠️ Перед приёмом любых препаратов рекомендуется проконсультироваться с врачом.\n\n"
            answer += "Возможно, вам подойдут следующие препараты:\n\n"
            for i, drug in enumerate(combined[:5], 1):
                # Очищаем название от латиницы и мусора
                clean_trade = re.sub(r'\(.*?\)', '', drug['trade_name'])
                clean_trade = re.sub(r'инструкция по применению', '', clean_trade, flags=re.IGNORECASE)
                clean_trade = re.sub(r'®', ' ', clean_trade)
                clean_trade = re.sub(r'\s+', ' ', clean_trade).strip()

                answer += f"**{i}. {clean_trade}**\n"
                if drug.get('indications'):
                    answer += f"   _Показания: {drug['indications'][:300]}..._\n"  # ← увеличили с 150 до 300
                answer += "\n"

            sources = [DrugSearchResult(id=r['id'], trade_name=r['trade_name'], source_url=r['source_url']) for r in
                       combined]
            return ChatResponse(answer=answer, sources=sources, type="symptom")

    except Exception as e:
        import traceback
        traceback.print_exc()
        return ChatResponse(answer=f"Ошибка сервера: {str(e)}", sources=[], type='error')


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)