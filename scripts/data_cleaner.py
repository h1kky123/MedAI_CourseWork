import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


class DataCleaner:

    def __init__(self, input_file: str, output_file: Optional[str] = None):
        self.input_file = Path(input_file)
        self.output_file = output_file or f"cleaned_{self.input_file.name}"
        self.data = []

    def load_data(self) -> List[Dict]:
        with open(self.input_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        return self.data

    def remove_duplicates(self) -> int:
        # Удаление дубликатов по URL и названию
        seen_urls = set()
        unique_data = []

        for item in self.data:
            url = item.get('source_url', '')
            name = item.get('name', '').strip()
            if url not in seen_urls and name:
                seen_urls.add(url)
                unique_data.append(item)

        self.data = unique_data
        return len(self.data)

    def clean_text(self, text: str) -> str:
        # Очистка текста от лишних символов, пробелов и HTML-сущностей
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('[:50]', '').replace('...', '')
        text = re.sub(r'<[^>]+>', '', text)
        text = text.replace('&nbsp;', ' ').replace('&mdash;', '—')
        text = text.replace('&ndash;', '–').replace('&laquo;', '«')
        text = text.replace('&raquo;', '»').replace('&quot;', '"')
        text = text.replace('&amp;', '&')
        return text.strip()

    def clean_all_records(self) -> int:
        # Очистка текстовых полей во всех записях
        text_fields = [
            'name', 'form', 'description', 'indications',
            'contraindications', 'side_effects', 'precautions',
            'usage_instructions'
        ]
        for item in self.data:
            for field in text_fields:
                if field in item:
                    item[field] = self.clean_text(item[field])
        return len(self.data)

    def filter_empty_records(self) -> int:
        # Удаление записей без названия или описания
        self.data = [
            item for item in self.data
            if item.get('name') and (item.get('description') or item.get('indications'))
        ]
        return len(self.data)

    def normalize_names(self) -> int:
        # Нормализация названий препаратов
        for item in self.data:
            name = item.get('name', '')
            name = re.sub(r'\s*инструкция по применению\s*', '', name, flags=re.IGNORECASE)
            name = re.sub(r'\s*\(\s*', ' (', name)
            name = re.sub(r'\s*\)\s*', ') ', name)
            item['name'] = name.strip()
        return len(self.data)

    def add_metadata(self):
        # Добавление метаданных: активное вещество, категория, дата очистки
        for item in self.data:
            item['active_substance'] = self._extract_active_substance(item.get('name', ''))
            item['category'] = self._determine_category(item)
            item['cleaned_at'] = datetime.now().isoformat()

    def _extract_active_substance(self, name: str) -> str:
        # Извлечение активного вещества из названия препарата
        match = re.search(r'\(([^)]+)\)', name)
        if match:
            return match.group(1).strip()
        parts = re.split(r'[\s(]', name)
        return parts[0] if parts else name

    def _determine_category(self, item: Dict) -> str:
        # Определение категории препарата по тексту описания
        description = (
            item.get('description', '') + ' ' +
            item.get('indications', '') + ' ' +
            item.get('usage_instructions', '')
        ).lower()

        categories = {
            'антибиотик': ['антибиотик', 'бактерицидн', 'бактериостатич'],
            'антигипертензивное': ['артериальн.*гипертензи', 'гипотензивн', 'антигипертензивн'],
            'обезболивающее': ['обезболивающ', 'анальгезирующ', 'болеутоляющ'],
            'антигистаминное': ['антигистаминн', 'аллергич', 'гистамин'],
            'противовоспалительное': ['противовоспалител', 'нпвс', 'нестероид'],
            'витамин': ['витамин', 'гиповитаминоз', 'авитаминоз'],
            'вакцина': ['вакцин', 'иммунизаци', 'профилактик.*прививк'],
        }

        for category, keywords in categories.items():
            for keyword in keywords:
                if re.search(keyword, description):
                    return category
        return 'другое'

    def clean_and_save(self) -> str:
        # Полный цикл очистки и сохранение результата в файл
        self.load_data()
        self.remove_duplicates()
        self.filter_empty_records()
        self.clean_all_records()
        self.normalize_names()
        self.add_metadata()

        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

        return self.output_file


def main():
    data_files = list(Path('..').glob('vidal_dataset_*.json'))
    if not data_files:
        return

    input_file = max(data_files, key=lambda f: f.stat().st_mtime)
    cleaner = DataCleaner(
        input_file=str(input_file),
        output_file='cleaned_medicines.json'
    )
    cleaner.clean_and_save()


if __name__ == '__main__':
    main()