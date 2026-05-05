"""
Обновленный модуль поиска аналогов препаратов
Использует новую структуру БД с активными веществами и АТХ кодами
"""
import os
from typing import List, Dict, Optional
from vector_store_v2 import PostgreSQLVectorStoreV2


class DrugAnalogsFinderV2:
    """Поиск аналогов с новой структурой БД"""
    
    def __init__(self, db_config: Optional[Dict] = None):
        self.vector_store = PostgreSQLVectorStoreV2(db_config)
        
    def connect(self):
        """Подключение к базе данных"""
        self.vector_store.connect()
    
    def close(self):
        """Закрытие соединения"""
        self.vector_store.close()
    
    def find_structural_analogs(
        self, 
        drug_id: int,
        min_common_substances: int = 1,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Поиск структурных аналогов (по активным веществам)
        
        Args:
            drug_id: ID препарата
            min_common_substances: Минимальное количество общих веществ
            top_k: Количество результатов
        """
        analogs = self.vector_store.get_analogs_by_substance(drug_id, top_k)
        
        # Фильтрация по минимальному количеству общих веществ
        return [
            a for a in analogs 
            if a['common_count'] >= min_common_substances
        ]
    
    def find_therapeutic_analogs(
        self,
        drug_id: int,
        min_common_atc: int = 1,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Поиск терапевтических аналогов (по АТХ кодам)
        
        Args:
            drug_id: ID препарата
            min_common_atc: Минимальное количество общих АТХ кодов
            top_k: Количество результатов
        """
        analogs = self.vector_store.get_analogs_by_atc(drug_id, top_k)
        
        # Фильтрация по минимальному количеству общих АТХ
        return [
            a for a in analogs 
            if a['common_atc_count'] >= min_common_atc
        ]
    
    def find_analogs_by_name(
        self,
        drug_name: str,
        top_k: int = 10
    ) -> Dict:
        """
        Поиск аналогов по названию препарата
        
        Returns:
            Dict с drug_info и analogs
        """
        # Находим препарат по названию
        drug_id = self._find_drug_id_by_name(drug_name)
        
        if not drug_id:
            return {'error': f'Препарат "{drug_name}" не найден'}
        
        # Получаем информацию о препарате
        drug_info = self.vector_store.get_drug_details(drug_id)
        
        # Ищем структурные аналоги
        structural_analogs = self.find_structural_analogs(drug_id, top_k=top_k)
        
        # Ищем терапевтические аналоги
        therapeutic_analogs = self.find_therapeutic_analogs(drug_id, top_k=top_k)
        
        return {
            'drug_info': drug_info,
            'structural_analogs': structural_analogs,
            'therapeutic_analogs': therapeutic_analogs
        }
    
    def find_by_indication(
        self,
        indication: str,
        top_k: int = 10
    ) -> List[Dict]:
        """Поиск препаратов по показанию"""
        # Полнотекстовый поиск по показаниям
        return self.vector_store.search_drugs_fulltext(indication, top_k)
    
    def find_by_substance(
        self,
        substance_name: str,
        top_k: int = 20
    ) -> List[Dict]:
        """Поиск всех препаратов с данным активным веществом"""
        return self.vector_store.search_by_active_substance(substance_name)
    
    def compare_drugs(self, drug_id1: int, drug_id2: int) -> Dict:
        """
        Сравнение двух препаратов
        
        Returns:
            Dict с информацией о сходстве
        """
        details1 = self.vector_store.get_drug_details(drug_id1)
        details2 = self.vector_store.get_drug_details(drug_id2)
        
        if not details1 or not details2:
            return {'error': 'Один или оба препарата не найдены'}
        
        # Общие активные вещества
        substances1 = set(s['name_ru'] for s in details1.get('substances', []))
        substances2 = set(s['name_ru'] for s in details2.get('substances', []))
        common_substances = substances1 & substances2
        
        # Общие АТХ коды
        atc1 = set(a['code'] for a in details1.get('atc_codes', []))
        atc2 = set(a['code'] for a in details2.get('atc_codes', []))
        common_atc = atc1 & atc2
        
        return {
            'drug1': {
                'name': details1['trade_name'],
                'manufacturer': details1.get('manufacturer', ''),
                'substances': list(substances1),
                'atc_codes': list(atc1)
            },
            'drug2': {
                'name': details2['trade_name'],
                'manufacturer': details2.get('manufacturer', ''),
                'substances': list(substances2),
                'atc_codes': list(atc2)
            },
            'comparison': {
                'common_substances': list(common_substances),
                'common_substance_count': len(common_substances),
                'common_atc_codes': list(common_atc),
                'common_atc_count': len(common_atc),
                'structural_similarity': len(common_substances) / max(len(substances1 | substances2), 1),
                'therapeutic_similarity': len(common_atc) / max(len(atc1 | atc2), 1)
            }
        }
    
    def get_best_analog(
        self,
        drug_id: int,
        criteria: str = 'structural'
    ) -> Optional[Dict]:
        """
        Поиск лучшего аналога
        
        Args:
            drug_id: ID препарата
            criteria: 'structural' или 'therapeutic'
        """
        if criteria == 'structural':
            analogs = self.find_structural_analogs(drug_id, top_k=1)
        else:
            analogs = self.find_therapeutic_analogs(drug_id, top_k=1)
        
        return analogs[0] if analogs else None
    
    def _find_drug_id_by_name(self, drug_name: str) -> Optional[int]:
        """Поиск ID препарата по названию"""
        try:
            with self.vector_store.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT id
                    FROM drugs
                    WHERE trade_name ILIKE %s
                    LIMIT 1;
                """, (f'%{drug_name}%',))
                
                result = cursor.fetchone()
                return result[0] if result else None
        
        except Exception as e:
            print(f"❌ Ошибка поиска ID: {e}")
            return None


def main():
    """Тестирование поиска аналогов v2"""
    print("=== Тестирование DrugAnalogsFinderV2 ===")
    
    finder = DrugAnalogsFinderV2()
    finder.connect()
    
    # Находим любой препарат с веществами
    print("\n1. Поиск препаратов с активными веществами...")
    substance_results = finder.find_by_substance("", top_k=1)
    
    # Тест с реальным препаратом
    print("\n2. Поиск аналогов по названию...")
    test_drug = input("Введите название препарата (или Enter для 'А-Валкордис'): ").strip()
    if not test_drug:
        test_drug = "А-Валкордис"
    
    result = finder.find_analogs_by_name(test_drug, top_k=5)
    
    if 'error' in result:
        print(f"❌ {result['error']}")
    else:
        drug_info = result['drug_info']
        print(f"\nПрепарат: {drug_info['trade_name']}")
        print(f"Производитель: {drug_info.get('manufacturer', 'Н/Д')}")
        print(f"Активные вещества:")
        for s in drug_info.get('substances', []):
            print(f"  - {s['name_ru']} ({s.get('name_en', '')})")
        
        print(f"\nСтруктурные аналоги ({len(result['structural_analogs'])}):")
        for a in result['structural_analogs']:
            print(f"  - {a['trade_name']} (общих веществ: {a['common_count']})")
        
        print(f"\nТерапевтические аналоги ({len(result['therapeutic_analogs'])}):")
        for a in result['therapeutic_analogs'][:5]:
            print(f"  - {a['trade_name']} (общих АТХ: {a['common_atc_count']})")
    
    finder.close()
    print("\n✓ Тестирование завершено!")


if __name__ == '__main__':
    main()
