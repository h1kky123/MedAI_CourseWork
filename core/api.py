import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import re
import torch
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pathlib
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from core.vector_store import PostgreSQLVectorStoreV2
from core.drug_analogs import DrugAnalogsFinderV2
from core.intent_predictor import IntentPredictor


# Модели данных

class DrugSearchResult(BaseModel):
    id: int
    trade_name: str
    source_url: str
    manufacturer: Optional[str] = None
    description: Optional[str] = None
    indications: Optional[str] = None


class ChatRequest(BaseModel):
    query: str
    top_k: int = 10


class ChatResponse(BaseModel):
    answer: str
    sources: List[DrugSearchResult] = []
    type: str


# Глобальные объекты

vector_store: Optional[PostgreSQLVectorStoreV2] = None
analog_finder: Optional[DrugAnalogsFinderV2] = None
intent_classifier: Optional[IntentPredictor] = None
tokenizer = None
model = None
model_device = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = pathlib.Path(__file__).parent.parent
STATIC_DIR = BASE_DIR / "static"
LORA_PATH = BASE_DIR / "/models/qwen_medical_lora"


# Инициализация приложения

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store, analog_finder, intent_classifier, tokenizer, model

    # Подключение к БД
    try:
        vector_store = PostgreSQLVectorStoreV2()
        vector_store.connect()
        analog_finder = DrugAnalogsFinderV2()
        analog_finder.connect()
    except Exception as e:
        raise e

    # Загрузка классификатора намерений
    try:
        intent_classifier = IntentPredictor()
    except Exception as e:
        intent_classifier = None

    # Загрузка Qwen + LoRA
    try:
        base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if model_device == "cuda" else torch.float32,
            device_map="auto" if model_device == "cuda" else None
        )

        model = PeftModel.from_pretrained(base_model, str(LORA_PATH)) if LORA_PATH.exists() else base_model

        if model_device == "cpu":
            model.to(model_device)
        model.eval()
    except Exception as e:
        raise e

    yield

    if vector_store: vector_store.close()
    if analog_finder: analog_finder.close()


# Приложение

app = FastAPI(title="MedAI RAG", version="5.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
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
    # Генерация ответа через Qwen + LoRA
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


def clean_drug_name(name: str) -> str:
    # Очистка названия препарата от латиницы, символа ® и мусора
    name = re.sub(r'\(.*?\)', '', name)
    name = re.sub(r'инструкция по применению', '', name, flags=re.IGNORECASE)
    name = re.sub(r'®', ' ', name)
    return re.sub(r'\s+', ' ', name).strip()


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not vector_store:
        raise HTTPException(status_code=503, detail="БД не подключена")

    try:
        query = request.query
        top_k = request.top_k

        # 1. Классификация намерения
        intent = "general"
        entity = None

        if intent_classifier:
            intent_result = intent_classifier.predict(query)
            intent = intent_result['intent']
            entity = intent_result['entity']
        else:
            # Fallback если классификатор не загружен
            query_lower = query.lower()
            if any(w in query_lower for w in ["аналог", "заменит", "похож"]):
                intent = "analogs"
            elif any(w in query_lower for w in ["инструкц", "как примен", "дозировк"]):
                intent = "drug_info"
            elif any(w in query_lower for w in ["что дел", "что прин", "болит", "боль"]):
                intent = "symptom"

        # Очистка сущности от стоп-слов
        stop_words_pattern = r'\b(инструкция|для|покажи|расскажи|про|о|что|такое|как)\b'
        if entity:
            entity_clean = re.sub(stop_words_pattern, '', entity, flags=re.IGNORECASE).strip()
            entity_clean = re.sub(r'[?!,;.)]', '', entity_clean).strip()
        else:
            entity_clean = query.strip()

        # 2. Поиск препарата в БД
        drug_id = None
        drug_name = None
        drug_details = None
        all_variants = []

        if entity and intent in ["analogs", "drug_info"]:
            words = [w for w in entity_clean.split() if len(w) > 1]

            with vector_store.connection.cursor() as c:
                if words:
                    query_parts = ["trade_name ILIKE %s" for _ in words]
                    params = [f'%{w}%' for w in words]
                    c.execute(f"SELECT id, trade_name FROM drugs WHERE {' AND '.join(query_parts)} LIMIT 20;", params)
                    all_variants = [{'id': r[0], 'name': r[1]} for r in c.fetchall()]

                if not all_variants and entity_clean:
                    c.execute("SELECT id, trade_name FROM drugs WHERE trade_name ILIKE %s LIMIT 20;",
                              (f'%{entity_clean}%',))
                    all_variants = [{'id': r[0], 'name': r[1]} for r in c.fetchall()]

                # Выбор лучшего совпадения из списка кандидатов
                if len(all_variants) > 1:
                    best_match = None
                    best_score = -1
                    clean_req_normalized = re.sub(r'\s+', '', entity_clean.lower())
                    req_words = set(entity_clean.lower().split())

                    for v in all_variants:
                        db_name_clean = re.sub(r'\(.*?\)', '', v['name'])
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

                        if score > best_score:
                            best_score = score
                            best_match = v

                    if best_match and best_score >= 10:
                        target_variant = best_match
                    else:
                        selection_text = f"Я нашел несколько препаратов, содержащих \"{entity_clean}\". Пожалуйста, уточните:\n\n"
                        for i, d in enumerate(all_variants[:10], 1):
                            selection_text += f"{i}. {d['name']}\n"
                        selection_text += "\nНапример, напишите: *Инструкция для Нурофен Экспресс*"
                        sources_obj = [DrugSearchResult(id=d['id'], trade_name=d['name'], source_url='') for d in all_variants[:10]]
                        return ChatResponse(answer=selection_text, sources=sources_obj, type="selection")

                elif len(all_variants) == 1:
                    target_variant = all_variants[0]
                else:
                    target_variant = None

                if target_variant:
                    drug_id = target_variant['id']
                    drug_name = target_variant['name']
                    drug_details = vector_store.get_drug_details(drug_id)

        # 3. Формирование ответа

        # Аналоги
        if intent == "analogs" and drug_id and drug_details and analog_finder:
            structural = analog_finder.find_structural_analogs(drug_id, top_k=10)
            therapeutic = analog_finder.find_therapeutic_analogs(drug_id, top_k=5)

            context = f"Препарат: {drug_details['trade_name']}\n"
            if drug_details.get('substances'):
                context += f"Активные вещества: {', '.join([s['name_ru'] for s in drug_details['substances']])}\n"
            if drug_details.get('indications'):
                context += f"Показания: {drug_details['indications'][:300]}\n"
            if structural:
                context += "\nСтруктурные аналоги:\n" + "".join([f"- {a['trade_name']}\n" for a in structural[:5]])
            if therapeutic:
                context += "\nТерапевтические аналоги:\n" + "".join([f"- {a['trade_name']}\n" for a in therapeutic[:5]])

            prompt = f"""Ты медицинский ассистент. Пользователь ищет аналоги препарата.
{context}
Задача: кратко объясни разницу между структурными и терапевтическими аналогами,
порекомендуй на что обратить внимание при выборе замены.
Напомни что окончательный выбор должен делать врач.
Отвечай на русском языке, кратко и по делу."""

            qwen_answer = generate_response_with_qwen(prompt, max_tokens=300)

            answer = f"**{clean_drug_name(drug_details['trade_name'])}**"
            if drug_details.get('substances'):
                subs = ", ".join([s['name_ru'] for s in drug_details['substances']])
                answer += f" — препарат на основе {subs}."
            if drug_details.get('indications'):
                answer += f"\n_{drug_details['indications'][:200]}..._"
            answer += "\n\n"

            if structural:
                answer += "**Структурные аналоги** (те же вещества):\n"
                for i, a in enumerate(structural[:10], 1):
                    subs = a.get('substances', [])
                    sub_str = f" ({', '.join(subs[:3])})" if subs else ""
                    answer += f"{i}. **{clean_drug_name(a['trade_name'])}**{sub_str}\n"

            if therapeutic:
                answer += "\n**Терапевтические аналоги** (похожее действие):\n"
                for i, a in enumerate(therapeutic[:5], 1):
                    answer += f"{i}. {clean_drug_name(a['trade_name'])}\n"

            if not structural and not therapeutic:
                answer += "Аналоги не найдены.\n"

            answer += f"\n---\n**Комментарий ИИ:**\n{qwen_answer}"

            sources = [DrugSearchResult(id=drug_id, trade_name=drug_name, source_url=drug_details.get('source_url', ''))]
            return ChatResponse(answer=answer, sources=sources, type="analogs")

        # Информация о препарате
        elif intent == "drug_info" and drug_id and drug_details:
            answer = f"**Информация о препарате {drug_details['trade_name']}:**\n\n"

            if drug_details.get('substances'):
                answer += f"**Активные вещества:** {', '.join([s['name_ru'] for s in drug_details['substances']])}\n\n"
            if drug_details.get('indications'):
                answer += f"**Показания:** {drug_details['indications'][:1000]}\n\n"
            if drug_details.get('contraindications'):
                answer += f"**Противопоказания:** {drug_details['contraindications'][:800]}\n\n"
            if drug_details.get('usage_instructions'):
                answer += f"**Применение:** {drug_details['usage_instructions'][:800]}\n\n"
            if drug_details.get('side_effects'):
                answer += f"**Побочные эффекты:** {drug_details['side_effects'][:600]}\n\n"

            other_variants = [v for v in all_variants if v['id'] != drug_id]
            if other_variants:
                answer += "\n**Другие формы этого препарата:**\n"
                for v in other_variants[:5]:
                    answer += f"• {clean_drug_name(v['name'])}\n"

            if drug_details.get('source_url'):
                answer += f"\n**Источник:** {drug_details['source_url']}\n\n"

            sources = [DrugSearchResult(id=drug_id, trade_name=drug_name, source_url=drug_details.get('source_url', ''))]
            return ChatResponse(answer=answer, sources=sources, type="drug_info")

        # Симптомы / общий поиск
        else:
            results_fts = vector_store.search_drugs_fulltext(query, top_k=top_k)
            results_sem = vector_store.search_drugs_semantic(query, top_k=top_k)

            # Фильтруем — оставляем только те где симптом упомянут в показаниях
            query_words = [w.lower() for w in query.split() if len(w) > 3]
            seen_ids = set()
            combined = []

            for r in results_fts + results_sem:
                if r['id'] in seen_ids:
                    continue
                indications = (r.get('indications') or '').lower()
                if any(w in indications for w in query_words):
                    seen_ids.add(r['id'])
                    combined.append(r)
                    if len(combined) >= 5:
                        break

            # Если после фильтрации ничего не осталось — берём без фильтра
            if not combined:
                for r in results_fts + results_sem:
                    if r['id'] not in seen_ids:
                        seen_ids.add(r['id'])
                        combined.append(r)
                        if len(combined) >= 5:
                            break

            if not combined:
                return ChatResponse(answer="К сожалению, ничего не найдено.", sources=[], type="general")

            context = ""
            for drug in combined[:5]:
                context += f"\nПрепарат: {clean_drug_name(drug['trade_name'])}\n"
                if drug.get('indications'):
                    context += f"Показания: {drug['indications'][:200]}\n"

            prompt = f"""Ты медицинский ассистент. Пользователь спрашивает: "{query}"
Вот препараты из базы данных, которые могут подойти:
{context}
Задача: объясни какие из этих препаратов и почему могут помочь при данном симптоме.
Обязательно напомни что необходимо проконсультироваться с врачом перед приёмом.
Отвечай на русском языке, кратко и понятно."""

            qwen_answer = generate_response_with_qwen(prompt, max_tokens=300)

            answer = "Перед приёмом любых препаратов проконсультируйтесь с врачом.\n\n"
            answer += "**Возможно подходящие препараты:**\n\n"
            for i, drug in enumerate(combined[:5], 1):
                answer += f"{i}. **{clean_drug_name(drug['trade_name'])}**\n"
                if drug.get('indications'):
                    answer += f"   _{drug['indications'][:300]}..._\n"
                answer += "\n"
            answer += f"---\n**Комментарий ИИ:**\n{qwen_answer}"

            sources = [DrugSearchResult(id=r['id'], trade_name=r['trade_name'], source_url=r['source_url']) for r in combined]
            return ChatResponse(answer=answer, sources=sources, type="symptom")

    except Exception as e:
        import traceback
        traceback.print_exc()
        return ChatResponse(answer=f"Ошибка сервера: {str(e)}", sources=[], type='error')


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("core.api:app", host="0.0.0.0", port=8000, reload=False)