# Структура базы данных - Гибридная схема

## Обзор

Гибридная структура combines нормализованные таблицы для справочных данных с JSONB для гибкого хранения контента.

## Схема данных

```
┌─────────────────────────┐
│    active_substances    │  ← Справочник активных веществ (MNN)
├─────────────────────────┤
│ • id (PK)               │
│ • name_ru (UNIQUE)      │  ← Русское название
│ • name_en               │  ← Международное название
│ • inchi_key             │  ← Химический идентификатор
│ • cas_number            │  ← CAS номер
└─────────────────────────┘
          ↑
          │ drug_substances (many-to-many)
          │
┌─────────────────────────┐         ┌─────────────────────────┐
│         drugs           │         │       atc_codes         │
├─────────────────────────┤         ├─────────────────────────┤
│ • id (PK)               │         │ • id (PK)               │
│ • trade_name            │         │ • atc_code (UNIQUE)     │
│ • source_url (UNIQUE)   │         │ • name                  │
│ • manufacturer          │         │ • level                 │
│ • reg_number            │         │ • parent_code           │
│ • prescription_status   │         └─────────────────────────┘
│ • created_at            │                    ↑
│ • updated_at            │         ┌──────────┴──────────────┐
└──────────┬──────────────┘         │      drug_atc           │
           │                        │ (many-to-many связь)    │
           │                        └─────────────────────────┘
           │
           ├────────────────────────┐
           │                        │
    ┌──────▼──────┐          ┌─────▼──────┐
    │ drug_forms  │          │drug_substances│
    ├─────────────┤          ├──────────────┤
    │ • id (PK)   │          │ • drug_id    │
    │ • drug_id   │          │ • substance_id│
    │ • form_desc │          │ • dosage     │
    │ • dosage    │          └──────────────┘
    │ • pkg_size  │
    └─────────────┘


┌─────────────────────────────────────────┐
│            drug_content                  │  ← Полнотекстовый контент
├─────────────────────────────────────────┤
│ • drug_id (PK, FK → drugs)              │
│ • description                           │
│ • indications                           │
│ • contraindications                     │
│ • side_effects                          │
│ • precautions                           │
│ • usage_instructions                    │
│ • interactions                          │
│ • overdose                              │
│ • special_instructions                  │
│ • storage_conditions                    │
│ • shelf_life                            │
│ • search_vector (TSVECTOR для поиска)   │
└─────────────────────────────────────────┘
```

## Преимущества гибридной схемы

### ✅ Нормализованные таблицы
- **active_substances** - справочник веществ, легко искать аналоги
- **atc_codes** - стандартная классификация ВОЗ
- **drug_forms** - множественные формы выпуска одного препарата
- **drug_substances** - связь многие-ко-многим (комбинированные препараты)

### ✅ Полнотекстовый поиск
- **drug_content** - отдельная таблица с TSVECTOR индексом
- Быстрый поиск по всем текстовым полям
- Не замедляет основные таблицы

### ✅ Гибкость
- JSONB можно добавить в drug_content для дополнительных полей
- Легко добавлять новые поля без изменения схемы

## SQL запросы для поиска аналогов

### Поиск по активному веществу
```sql
SELECT d.trade_name, d.manufacturer, s.name_ru, s.name_en, ds.dosage
FROM drugs d
JOIN drug_substances ds ON d.id = ds.drug_id
JOIN active_substances s ON ds.substance_id = s.id
WHERE s.name_ru = 'ибупрофен';
```

### Поиск комбинированных препаратов
```sql
SELECT d.trade_name, array_agg(s.name_ru) as substances
FROM drugs d
JOIN drug_substances ds ON d.id = ds.drug_id
JOIN active_substances s ON ds.substance_id = s.id
GROUP BY d.id
HAVING COUNT(ds.substance_id) > 1;
```

### Поиск по АТХ коду
```sql
SELECT d.trade_name, a.atc_code, a.name
FROM drugs d
JOIN drug_atc da ON d.id = da.drug_id
JOIN atc_codes a ON da.atc_id = a.id
WHERE a.atc_code LIKE 'M01A%';  -- НПВС
```

### Полнотекстовый поиск
```sql
SELECT d.trade_name, dc.indications, dc.description
FROM drugs d
JOIN drug_content dc ON d.id = dc.drug_id
WHERE dc.search_vector @@ to_tsquery('russian', 'головная & боль')
ORDER BY ts_rank(dc.search_vector, to_tsquery('russian', 'головная & боль')) DESC;
```

## Индексы

```sql
-- Полнотекстовый индекс
CREATE INDEX idx_drug_content_search ON drug_content USING GIN(search_vector);

-- Индексы для внешних ключей
CREATE INDEX idx_drug_forms_drug_id ON drug_forms(drug_id);
CREATE INDEX idx_drug_substances_drug_id ON drug_substances(drug_id);
CREATE INDEX idx_drug_substances_substance_id ON drug_substances(substance_id);
CREATE INDEX idx_drug_atc_drug_id ON drug_atc(drug_id);
CREATE INDEX idx_drug_atc_atc_id ON drug_atc(atc_id);

-- Индекс для быстрого поиска по названию
CREATE INDEX idx_drugs_trade_name ON drugs(trade_name);
CREATE INDEX idx_active_substances_name_ru ON active_substances(name_ru);
```

## Миграция со старой схемы

Для миграции из старой таблицы `medicines` в новую структуру:

1. Создать новые таблицы
2. Перенести данные из `medicines` в `drugs` + `drug_content`
3. Извлечь активные вещества из текста (если возможно)
4. Обновить все зависимые модули

Скрипт миграции: `migrate_database.py`
