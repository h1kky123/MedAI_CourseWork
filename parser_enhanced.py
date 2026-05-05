"""
Парсер сайта vidal.ru - сбор полной фарм-карточки препаратов
Собирает: активные вещества, дозировки, производителя, рег.номер, клинические данные
"""
import requests
import time
import json
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Optional


START_URL = "https://www.vidal.ru/drugs/products/o/rus-a"

DB_CONFIG = {
    "dbname": "medicines_db",
    "user": "postgres",
    "password": "bkmzrjh1231",
    "host": "localhost",
    "port": "5433"
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "ru-RU,ru;q=0.9"
}

CYRILLIC_TO_LATIN = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ж": "zh", "з": "z",
    "и": "i", "й": "j", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p",
    "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "ts", "ч": "ch",
    "ш": "sh", "э": "eh", "ю": "yu", "я": "ya"
}


# ============================================================================
# БАЗА ДАННЫХ
# ============================================================================

def get_db_connection():
    import psycopg2
    return psycopg2.connect(**DB_CONFIG)


def create_tables():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS active_substances (
            id SERIAL PRIMARY KEY,
            name_ru TEXT NOT NULL UNIQUE,
            name_en TEXT
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS drugs (
            id SERIAL PRIMARY KEY,
            trade_name TEXT NOT NULL,
            source_url TEXT UNIQUE NOT NULL,
            manufacturer TEXT,
            reg_number TEXT,
            prescription_status TEXT,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS drug_forms (
            id SERIAL PRIMARY KEY,
            drug_id INTEGER REFERENCES drugs(id) ON DELETE CASCADE,
            form_description TEXT NOT NULL,
            dosage TEXT,
            package_size TEXT
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS drug_substances (
            drug_id INTEGER REFERENCES drugs(id) ON DELETE CASCADE,
            substance_id INTEGER REFERENCES active_substances(id) ON DELETE CASCADE,
            dosage TEXT,
            PRIMARY KEY (drug_id, substance_id)
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS drug_content (
            drug_id INTEGER PRIMARY KEY REFERENCES drugs(id) ON DELETE CASCADE,
            description TEXT,
            indications TEXT,
            contraindications TEXT,
            side_effects TEXT,
            precautions TEXT,
            usage_instructions TEXT,
            interactions TEXT,
            special_instructions TEXT,
            storage_conditions TEXT,
            search_vector TSVECTOR
        );
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_drug_content_search
        ON drug_content USING GIN(search_vector);
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("✓ Таблицы созданы/проверены")


# ============================================================================
# СБОР ССЫЛОК
# ============================================================================

def get_total_pages(soup):
    pagination = soup.find("div", class_="pagination")
    if not pagination:
        return 1
    max_page = 1
    for a in pagination.find_all("a", href=True):
        href = a["href"]
        if "?p=" in href:
            try:
                page_num = int(href.split("?p=")[1])
                if page_num > max_page:
                    max_page = page_num
            except (ValueError, IndexError):
                continue
    return max_page


def get_all_drug_links_for_letter(base_url):
    response = requests.get(base_url, headers=HEADERS, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")

    total_pages = get_total_pages(soup)
    print(f"  Страниц: {total_pages}")

    all_links = []
    for page in range(1, total_pages + 1):
        page_url = base_url if page == 1 else f"{base_url}?p={page}"
        print(f"  -> {page}", end="", flush=True)

        response = requests.get(page_url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")

        table = soup.find("table")
        if not table:
            continue

        for a in table.select("a[href^='/drugs/']"):
            href = a["href"].strip()
            if any(kw in href for kw in [
                "/drugs/products", "/drugs/interaction", "/drugs/molecules",
                "/drugs/clinic", "/drugs/pharm", "/drugs/atc",
                "/drugs/companies", "/drugs/disease", "/drugs/firm/",
                "/drugs/company/", "/drugs/nosology"
            ]):
                continue
            if href.count("/") == 2:
                all_links.append("https://www.vidal.ru" + href)

        time.sleep(0.5)

    unique_links = list(dict.fromkeys(all_links))
    print(f"\n  Найдено: {len(unique_links)}")
    return unique_links


# ============================================================================
# ПАРСИНГ
# ============================================================================

def extract_active_substances(soup):
    substances = []

    # Метод 1: Секция "Активные вещества"
    for heading in soup.find_all(['h2', 'h3']):
        heading_text = heading.get_text(' ', strip=True)
        if 'активн' in heading_text.lower() or 'действующ' in heading_text.lower():
            for sibling in heading.find_next_siblings():
                if sibling.name in ['ul', 'ol']:
                    for li in sibling.find_all('li'):
                        text = li.get_text(' ', strip=True)
                        match = re.match(r'([а-яА-ЯёЁ\-]+)\s*\(([a-zA-Z\s\-]+)\)', text)
                        if match:
                            ru_name = match.group(1).strip()
                            en_name = match.group(2).strip().split()[0]
                            if len(ru_name) > 2 and len(en_name) > 2:
                                substances.append({'name_ru': ru_name, 'name_en': en_name})
                    break
                if sibling.name in ['h2', 'h3']:
                    break
            if substances:
                return substances

    # Метод 2: Секция "Состав" с таблицей
    for heading in soup.find_all(['h2', 'h3']):
        if 'состав' in heading.get_text(' ', strip=True).lower():
            comp_div = heading.find_next_sibling('div', class_='composition')
            if comp_div:
                for td in comp_div.find_all('td'):
                    text = td.get_text(' ', strip=True)
                    match = re.match(r'([а-яА-ЯёЁ\-]+)\s*\(в форме', text)
                    if match:
                        substances.append({'name_ru': match.group(1).strip(), 'name_en': ''})
            if substances:
                return substances

    # Метод 3: Паттерн "русское (english) ... INN/Rec" по всей странице
    all_text = soup.get_text()
    for match in re.finditer(r'([а-яА-ЯёЁ\-]{3,})\s*\(([a-zA-Z\s\-]{3,})\)', all_text):
        ru_name = match.group(1).strip()
        en_name = match.group(2).strip().split()[0]
        start = max(0, match.start() - 100)
        end = min(len(all_text), match.end() + 100)
        context = all_text[start:end].lower()
        if any(kw in context for kw in ['inn', 'мнн', 'rec.', 'registered', 'зарегистрир']):
            if len(ru_name) > 2 and len(en_name) > 2:
                substances.append({'name_ru': ru_name, 'name_en': en_name})

    if substances:
        seen = set()
        unique = []
        for s in substances:
            key = s['name_ru'].lower() or s['name_en'].lower()
            if key not in seen:
                seen.add(key)
                unique.append(s)
        return unique

    # Метод 4: Из H1
    h1 = soup.find('h1')
    if h1:
        title = h1.get_text(' ', strip=True)
        match = re.search(r'\(([а-яА-ЯёЁA-Za-z\s\-]+)\)', title)
        if match:
            inside = match.group(1).strip()
            if re.match(r'[A-Za-z]', inside):
                return [{'name_ru': '', 'name_en': inside}]
            else:
                return [{'name_ru': inside, 'name_en': ''}]

    return []


def extract_dosages(soup):
    dosages = []

    # Метод 1: "Лекарственные формы" с паттерном X мг+Y мг
    h2_forms = soup.find('h2', string=re.compile(r'лекарственн.*форм', re.I))
    if h2_forms:
        forms_div = h2_forms.find_next_sibling('div')
        if forms_div:
            text = forms_div.get_text(' ', strip=True)
            for dosage in re.findall(r'(\d+\s*мг\+\d+\s*мг)', text):
                dosages.append({'dosage': dosage.strip(), 'package_size': ''})

    # Метод 2: "Лекарственная форма" с таблицей products-table
    if not dosages:
        h2_form = soup.find('h2', string=re.compile(r'лекарственн.*форма[^ы]', re.I))
        if h2_form:
            forms_div = h2_form.find_next_sibling('div', class_='block-content')
            if forms_div:
                table = forms_div.find('table', class_=re.compile(r'products-table', re.I))
                if table:
                    text = table.get_text(' ', strip=True)
                    dosage_match = re.search(r'(\d+\s*мг)', text)
                    if dosage_match:
                        dosages.append({'dosage': dosage_match.group(1).strip(), 'package_size': ''})
                    pkg_match = re.search(r'(\d+(?:\s*(?:или|,)\s*\d+)*)\s*шт', text)
                    if pkg_match and dosages:
                        dosages[0]['package_size'] = pkg_match.group(1) + ' шт'

    # Метод 3: Meta description
    if not dosages:
        meta = soup.find('meta', attrs={'name': 'description'})
        if meta:
            dosage_match = re.search(r'(\d+\s*мг)', meta.get('content', ''))
            if dosage_match:
                dosages.append({'dosage': dosage_match.group(1).strip(), 'package_size': ''})

    return dosages


def extract_manufacturer(soup):
    h2_owner = soup.find('h2', string=re.compile(r'владелец.*регистрац', re.I))
    if h2_owner:
        owner_div = h2_owner.find_next_sibling('div')
        if owner_div:
            return owner_div.get_text(' ', strip=True)
    div_owners = soup.find('div', class_='owners')
    if div_owners:
        return div_owners.get_text(' ', strip=True)
    return ""


def extract_reg_number(soup):
    h2_forms = soup.find('h2', string=re.compile(r'лекарственн.*форм', re.I))
    if h2_forms:
        forms_div = h2_forms.find_next_sibling('div')
        if forms_div:
            text = forms_div.get_text(' ', strip=True)
            match = re.search(r'(ЛП-№[^\s,]+)', text)
            if match:
                return match.group(1)
    return ""


def extract_prescription_status(soup):
    if soup.find(string=re.compile(r'по рецепту', re.I)):
        return "prescription"
    elif soup.find(string=re.compile(r'без рецепта', re.I)):
        return "otc"
    return ""


def extract_section_content(soup, keywords):
    for keyword in keywords:
        keyword_lower = keyword.lower()
        for tag in soup.find_all(['h2', 'h3', 'h4']):
            if keyword_lower in tag.get_text(' ', strip=True).lower():
                content = []
                for sibling in tag.find_next_siblings():
                    if sibling.name in ['h2', 'h3', 'h4']:
                        break
                    t = sibling.get_text(' ', strip=True)
                    if t:
                        content.append(t)
                return ' '.join(content) if content else ""
        for div in soup.find_all('div', class_=re.compile(r'block-content', re.I)):
            div_text = div.get_text(' ', strip=True)
            if div_text and keyword_lower in div_text.lower():
                return div_text
    return ""


def parse_drug_page(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")

        name_tag = soup.select_one("h1")
        if not name_tag or "Взаимодействие" in name_tag.get_text():
            return None

        trade_name = name_tag.get_text(strip=True)

        return {
            "trade_name": trade_name,
            "source_url": url.strip(),
            "manufacturer": extract_manufacturer(soup),
            "reg_number": extract_reg_number(soup),
            "prescription_status": extract_prescription_status(soup),
            "active_substances": extract_active_substances(soup),
            "dosages": extract_dosages(soup),
            "description": extract_section_content(soup, ["Фармакологическое действие", "Свойства"]),
            "indications": extract_section_content(soup, ["Показания"]),
            "contraindications": extract_section_content(soup, ["Противопоказания"]),
            "side_effects": extract_section_content(soup, ["Побочное действие"]),
            "precautions": extract_section_content(soup, ["Особые указания", "С осторожностью"]),
            "usage_instructions": extract_section_content(soup, ["Режим дозирования", "Применение"]),
            "interactions": extract_section_content(soup, ["Взаимодействие"]),
            "special_instructions": extract_section_content(soup, ["Особые указания"]),
            "storage_conditions": extract_section_content(soup, ["Условия хранения"])
        }
    except Exception as e:
        print(f"\n  Ошибка {url}: {e}")
        return None


# ============================================================================
# ДЕДУПЛИКАЦИЯ
# ============================================================================

def normalize_drug_name(name):
    name = re.sub(r'\s*инструкция по применению\s*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*\(описание\)\s*', '', name, flags=re.IGNORECASE)
    name = name.replace('®', '').replace('™', '')
    match = re.match(r'([а-яА-ЯёЁA-Za-z\s\-]+)', name)
    if match:
        return re.sub(r'\b[а-яё]{1,2}\b', '', match.group(1).strip().lower()).strip()
    return name.strip().lower()


def deduplicate_drugs(dataset):
    grouped = {}
    for drug in dataset:
        norm_name = normalize_drug_name(drug.get('trade_name', ''))
        grouped.setdefault(norm_name, []).append(drug)

    deduplicated = []
    removed = 0

    for norm_name, drugs in grouped.items():
        if len(drugs) == 1:
            deduplicated.append(drugs[0])
        else:
            best = max(drugs, key=lambda d: sum(1 for v in d.values() if v))

            # Объединяем вещества
            seen_sub, all_sub = set(), []
            for drug in drugs:
                for s in drug.get('active_substances', []):
                    key = s.get('name_ru', '').lower()
                    if key and key not in seen_sub:
                        seen_sub.add(key)
                        all_sub.append(s)
            best['active_substances'] = all_sub

            # Объединяем дозировки
            seen_doz, all_doz = set(), []
            for drug in drugs:
                for d in drug.get('dosages', []):
                    key = d.get('dosage', '')
                    if key and key not in seen_doz:
                        seen_doz.add(key)
                        all_doz.append(d)
            best['dosages'] = all_doz

            # Заполняем пустые поля
            for field in ['description', 'indications', 'contraindications',
                         'side_effects', 'precautions', 'usage_instructions',
                         'interactions', 'special_instructions', 'storage_conditions',
                         'manufacturer', 'reg_number']:
                if not best.get(field):
                    for drug in drugs:
                        if drug.get(field):
                            best[field] = drug[field]
                            break

            deduplicated.append(best)
            removed += len(drugs) - 1

    print(f"  Дедупликация: {len(dataset)} → {len(deduplicated)} (удалено {removed})")
    return deduplicated


# ============================================================================
# СОХРАНЕНИЕ
# ============================================================================

def save_to_json(data_list):
    filename = f"vidal_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)
    print(f"✓ Сохранено в JSON: {filename}")
    return filename


def save_to_db(data_list):
    if not data_list:
        return

    conn = get_db_connection()
    cur = conn.cursor()
    drugs_saved = substances_saved = 0

    for drug_data in tqdm(data_list, desc="Сохранение в БД"):
        try:
            cur.execute("""
                INSERT INTO drugs (trade_name, source_url, manufacturer, reg_number, prescription_status)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (source_url) DO UPDATE SET
                    trade_name = EXCLUDED.trade_name,
                    manufacturer = EXCLUDED.manufacturer,
                    reg_number = EXCLUDED.reg_number,
                    prescription_status = EXCLUDED.prescription_status,
                    updated_at = NOW()
                RETURNING id;
            """, (
                drug_data.get('trade_name', ''),
                drug_data.get('source_url', ''),
                drug_data.get('manufacturer', ''),
                drug_data.get('reg_number', ''),
                drug_data.get('prescription_status', '')
            ))

            drug_id = cur.fetchone()[0]
            drugs_saved += 1

            for substance in drug_data.get('active_substances', []):
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

            for dosage_info in drug_data.get('dosages', []):
                cur.execute("""
                    INSERT INTO drug_forms (drug_id, form_description, dosage, package_size)
                    VALUES (%s, %s, %s, %s);
                """, (
                    drug_id,
                    drug_data.get('trade_name', ''),
                    dosage_info.get('dosage', ''),
                    dosage_info.get('package_size', '')
                ))

            cur.execute("""
                INSERT INTO drug_content (
                    drug_id, description, indications, contraindications,
                    side_effects, precautions, usage_instructions,
                    interactions, special_instructions, storage_conditions
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (drug_id) DO UPDATE SET
                    description = EXCLUDED.description,
                    indications = EXCLUDED.indications,
                    contraindications = EXCLUDED.contraindications,
                    side_effects = EXCLUDED.side_effects,
                    precautions = EXCLUDED.precautions,
                    usage_instructions = EXCLUDED.usage_instructions,
                    interactions = EXCLUDED.interactions,
                    special_instructions = EXCLUDED.special_instructions,
                    storage_conditions = EXCLUDED.storage_conditions;
            """, (
                drug_id,
                drug_data.get('description', ''),
                drug_data.get('indications', ''),
                drug_data.get('contraindications', ''),
                drug_data.get('side_effects', ''),
                drug_data.get('precautions', ''),
                drug_data.get('usage_instructions', ''),
                drug_data.get('interactions', ''),
                drug_data.get('special_instructions', ''),
                drug_data.get('storage_conditions', '')
            ))

        except Exception as e:
            print(f"\n  Ошибка сохранения {drug_data.get('trade_name')}: {e}")
            conn.rollback()
            continue

    conn.commit()
    cur.close()
    conn.close()
    print(f"✓ В БД сохранено: {drugs_saved} препаратов, {substances_saved} веществ")


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("="*60)
    print("ПАРСИНГ VIDAL.RU - ПОЛНАЯ ФАРМ-КАРТОЧКА")
    print("="*60)

    # 1. Создаем таблицы
    print("\n1. Создание/проверка таблиц БД...")
    create_tables()

    # 2. Сбор ссылок
    print("\n2. Сбор ссылок на препараты...")
    drug_links = []

    for cyrillic, latin in CYRILLIC_TO_LATIN.items():
        url = f"https://www.vidal.ru/drugs/products/o/rus-{latin}"
        print(f"\nБуква '{cyrillic.upper()}'")
        try:
            links = get_all_drug_links_for_letter(url)
            drug_links.extend(links)
            time.sleep(1)
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            continue

    # Удаление дубликатов ссылок
    drug_links = list(dict.fromkeys(drug_links))
    print(f"\n{'='*60}")
    print(f"Уникальных ссылок: {len(drug_links)}")

    # 3. Парсинг препаратов
    print(f"\n3. Парсинг препаратов...")
    dataset = []

    for url in tqdm(drug_links, desc="Парсинг"):
        time.sleep(0.5)
        parsed = parse_drug_page(url)
        if parsed:
            dataset.append(parsed)

    print(f"\n{'='*60}")
    print(f"Собрано препаратов: {len(dataset)}")

    # 4. Дедупликация
    dataset = deduplicate_drugs(dataset)

    # 5. Сохранение
    print(f"\n4. Сохранение данных...")
    save_to_json(dataset)
    save_to_db(dataset)

    print(f"\n{'='*60}")
    print("✓ ПАРСИНГ ЗАВЕРШЁН!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
