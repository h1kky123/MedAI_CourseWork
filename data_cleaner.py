"""
Модуль очистки и предобработки данных о лекарственных препаратах
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


class DataCleaner:
    """Очистка и структурирование данных о лекарствах"""
    
    def __init__(self, input_file: str, output_file: Optional[str] = None):
        self.input_file = Path(input_file)
        self.output_file = output_file or f"cleaned_{self.input_file.name}"
        self.data = []
        
    def load_data(self) -> List[Dict]:
        """Загрузка данных из JSON файла"""
        print(f"Загрузка данных из {self.input_file}...")
        with open(self.input_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"Загружено {len(self.data)} записей")
        return self.data
    
    def remove_duplicates(self) -> int:
        """Удаление дубликатов по URL и названию"""
        seen_urls = set()
        unique_data = []
        duplicates = 0
        
        for item in self.data:
            url = item.get('source_url', '')
            name = item.get('name', '').strip()
            
            if url not in seen_urls and name:
                seen_urls.add(url)
                unique_data.append(item)
            else:
                duplicates += 1
        
        print(f"Удалено дубликатов: {duplicates}")
        self.data = unique_data
        return duplicates
    
    def clean_text(self, text: str) -> str:
        """Очистка текста от лишних символов и пробелов"""
        if not text:
            return ""
        
        # Удаление множественных пробелов
        text = re.sub(r'\s+', ' ', text)
        
        # Удаление специальных символов парсинга
        text = text.replace('[:50]', '').replace('...', '')
        
        # Очистка от HTML-сущностей
        text = re.sub(r'<[^>]+>', '', text)
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&mdash;', '—')
        text = text.replace('&ndash;', '–')
        text = text.replace('&laquo;', '«')
        text = text.replace('&raquo;', '»')
        text = text.replace('&quot;', '"')
        text = text.replace('&amp;', '&')
        
        # Удаление пробелов в начале и конце
        text = text.strip()
        
        return text
    
    def clean_all_records(self) -> int:
        """Очистка всех записей"""
        cleaned_count = 0
        
        for item in self.data:
            # Очистка всех текстовых полей
            text_fields = [
                'name', 'form', 'description', 'indications',
                'contraindications', 'side_effects', 'precautions',
                'usage_instructions'
            ]
            
            for field in text_fields:
                if field in item:
                    item[field] = self.clean_text(item[field])
            
            cleaned_count += 1
        
        print(f"Очищено записей: {cleaned_count}")
        return cleaned_count
    
    def filter_empty_records(self) -> int:
        """Удаление записей без названия или описания"""
        initial_count = len(self.data)
        
        self.data = [
            item for item in self.data
            if item.get('name') and (item.get('description') or item.get('indications'))
        ]
        
        removed = initial_count - len(self.data)
        print(f"Удалено пустых записей: {removed}")
        return removed
    
    def normalize_names(self) -> int:
        """Нормализация названий препаратов"""
        normalized_count = 0
        
        for item in self.data:
            name = item.get('name', '')
            
            # Удаление "инструкция по применению" из названия
            name = re.sub(r'\s*инструкция по применению\s*', '', name, flags=re.IGNORECASE)
            
            # Удаление лишних пробелов вокруг скобок
            name = re.sub(r'\s*\(\s*', ' (', name)
            name = re.sub(r'\s*\)\s*', ') ', name)
            name = name.strip()
            
            item['name'] = name
            normalized_count += 1
        
        print(f"Нормализовано названий: {normalized_count}")
        return normalized_count
    
    def add_metadata(self):
        """Добавление метаданных к записям"""
        for item in self.data:
            # Извлечение активного вещества из названия
            item['active_substance'] = self._extract_active_substance(item.get('name', ''))
            
            # Определение категории препарата
            item['category'] = self._determine_category(item)
            
            # Добавление даты очистки
            item['cleaned_at'] = datetime.now().isoformat()
    
    def _extract_active_substance(self, name: str) -> str:
        """Извлечение активного вещества из названия"""
        # Попытка извлечь название из скобок
        match = re.search(r'\(([^)]+)\)', name)
        if match:
            return match.group(1).strip()
        
        # Возврат первой части названия до скобок
        parts = re.split(r'[\s(]', name)
        return parts[0] if parts else name
    
    def _determine_category(self, item: Dict) -> str:
        """Определение категории препарата по описанию"""
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
    
    def get_statistics(self) -> Dict:
        """Получение статистики по данным"""
        stats = {
            'total_records': len(self.data),
            'records_with_description': 0,
            'records_with_indications': 0,
            'records_with_side_effects': 0,
            'avg_description_length': 0,
            'categories': {},
        }
        
        total_desc_length = 0
        
        for item in self.data:
            if item.get('description'):
                stats['records_with_description'] += 1
                total_desc_length += len(item['description'])
            
            if item.get('indications'):
                stats['records_with_indications'] += 1
            
            if item.get('side_effects'):
                stats['records_with_side_effects'] += 1
            
            category = item.get('category', 'другое')
            stats['categories'][category] = stats['categories'].get(category, 0) + 1
        
        if stats['records_with_description'] > 0:
            stats['avg_description_length'] = (
                total_desc_length // stats['records_with_description']
            )
        
        return stats
    
    def clean_and_save(self) -> str:
        """Полный цикл очистки и сохранение"""
        # Загрузка данных
        self.load_data()
        
        # Очистка
        print("\n=== Начало очистки данных ===")
        self.remove_duplicates()
        self.filter_empty_records()
        self.clean_all_records()
        self.normalize_names()
        self.add_metadata()
        
        # Статистика
        stats = self.get_statistics()
        print("\n=== Статистика ===")
        print(f"Всего записей: {stats['total_records']}")
        print(f"С описанием: {stats['records_with_description']}")
        print(f"С показаниями: {stats['records_with_indications']}")
        print(f"С побочными эффектами: {stats['records_with_side_effects']}")
        print(f"Категории: {stats['categories']}")
        
        # Сохранение
        print(f"\nСохранение в {self.output_file}...")
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        
        print("✓ Очистка завершена!")
        return self.output_file


def main():
    """Основная функция"""
    # Поиск последнего файла с данными
    data_files = list(Path('.').glob('vidal_dataset_*.json'))
    
    if not data_files:
        print("❌ Файлы данных не найдены!")
        return
    
    # Выбор последнего файла
    input_file = max(data_files, key=lambda f: f.stat().st_mtime)
    print(f"Используем файл: {input_file}")
    
    # Создание и запуск очистителя
    cleaner = DataCleaner(
        input_file=str(input_file),
        output_file='cleaned_medicines.json'
    )
    
    cleaner.clean_and_save()


if __name__ == '__main__':
    main()
