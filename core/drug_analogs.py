from typing import List, Dict, Optional
from core.vector_store import PostgreSQLVectorStoreV2


class DrugAnalogsFinderV2:
    def __init__(self, db_config: Optional[Dict] = None):
        self.vector_store = PostgreSQLVectorStoreV2(db_config)

    def connect(self):
        self.vector_store.connect()

    def close(self):
        self.vector_store.close()

    def find_structural_analogs(self, drug_id: int, min_common_substances: int = 1, top_k: int = 10) -> List[Dict]:
        # Поиск аналогов с теми же активными веществами
        analogs = self.vector_store.get_analogs_by_substance(drug_id, top_k)
        return [a for a in analogs if a['common_count'] >= min_common_substances]

    def find_therapeutic_analogs(self, drug_id: int, min_common_atc: int = 1, top_k: int = 10) -> List[Dict]:
        # Поиск аналогов со схожим терапевтическим действием
        analogs = self.vector_store.get_analogs_by_atc(drug_id, top_k)
        return [a for a in analogs if a['common_atc_count'] >= min_common_atc]

    def find_analogs_by_name(self, drug_name: str, top_k: int = 10) -> Dict:
        # Поиск аналогов по названию препарата
        drug_id = self._find_drug_id_by_name(drug_name)
        if not drug_id:
            return {'error': f'Препарат "{drug_name}" не найден'}

        return {
            'drug_info': self.vector_store.get_drug_details(drug_id),
            'structural_analogs': self.find_structural_analogs(drug_id, top_k=top_k),
            'therapeutic_analogs': self.find_therapeutic_analogs(drug_id, top_k=top_k)
        }

    def find_by_indication(self, indication: str, top_k: int = 10) -> List[Dict]:
        # Поиск препаратов по показанию
        return self.vector_store.search_drugs_fulltext(indication, top_k)

    def find_by_substance(self, substance_name: str, top_k: int = 20) -> List[Dict]:
        # Поиск препаратов по активному веществу
        return self.vector_store.search_by_active_substance(substance_name)

    def compare_drugs(self, drug_id1: int, drug_id2: int) -> Dict:
        # Сравнение двух препаратов по веществам и АТХ кодам
        details1 = self.vector_store.get_drug_details(drug_id1)
        details2 = self.vector_store.get_drug_details(drug_id2)

        if not details1 or not details2:
            return {'error': 'Один или оба препарата не найдены'}

        substances1 = set(s['name_ru'] for s in details1.get('substances', []))
        substances2 = set(s['name_ru'] for s in details2.get('substances', []))
        common_substances = substances1 & substances2

        atc1 = set(a['code'] for a in details1.get('atc_codes', []))
        atc2 = set(a['code'] for a in details2.get('atc_codes', []))
        common_atc = atc1 & atc2

        return {
            'drug1': {'name': details1['trade_name'], 'manufacturer': details1.get('manufacturer', ''),
                      'substances': list(substances1), 'atc_codes': list(atc1)},
            'drug2': {'name': details2['trade_name'], 'manufacturer': details2.get('manufacturer', ''),
                      'substances': list(substances2), 'atc_codes': list(atc2)},
            'comparison': {
                'common_substances': list(common_substances),
                'common_substance_count': len(common_substances),
                'common_atc_codes': list(common_atc),
                'common_atc_count': len(common_atc),
                'structural_similarity': len(common_substances) / max(len(substances1 | substances2), 1),
                'therapeutic_similarity': len(common_atc) / max(len(atc1 | atc2), 1)
            }
        }

    def get_best_analog(self, drug_id: int, criteria: str = 'structural') -> Optional[Dict]:
        # Поиск наилучшего аналога по критерию
        analogs = self.find_structural_analogs(drug_id, top_k=1) if criteria == 'structural' \
            else self.find_therapeutic_analogs(drug_id, top_k=1)
        return analogs[0] if analogs else None

    def _find_drug_id_by_name(self, drug_name: str) -> Optional[int]:
        # Поиск ID препарата по названию в БД
        try:
            with self.vector_store.connection.cursor() as cursor:
                cursor.execute("SELECT id FROM drugs WHERE trade_name ILIKE %s LIMIT 1;", (f'%{drug_name}%',))
                result = cursor.fetchone()
                return result[0] if result else None
        except Exception:
            return None