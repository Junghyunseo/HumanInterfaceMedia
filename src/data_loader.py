import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        try:
            self.data = pd.read_excel(self.file_path)
            print("Data loaded successfully!")
            print("Available columns:", self.data.columns)
            print("\nFirst few rows:")
            print(self.data.head())
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def preprocess_data(self):
        if self.data is None:
            self.load_data()
        
        processed_data = self.data[['member_id', 'card_count', 'card_type', 'wrong_count']].copy()
        processed_data = processed_data.dropna()  # 누락 데이터 제거
        processed_data = processed_data[processed_data['card_count'] > 0]
        processed_data = processed_data[processed_data['wrong_count'] >= 0]

        # 보드 크기 매핑
        processed_data['board_size'] = processed_data['card_count'].map({
            9: '3x3',
            16: '4x4',
            25: '5x5'
        })

        # 카드 타입 수동 매핑
        card_type_mapping = {"ALPHABET": 0, "SHAPE": 1, "NUMBER": 2, "ARABIC": 3}
        processed_data['card_type_encoded'] = processed_data['card_type'].map(card_type_mapping)

        # 에러율 계산
        processed_data['error_rate'] = (processed_data['wrong_count'] / processed_data['card_count']) * 100

        # 조합 컬럼 생성
        processed_data['combination'] = processed_data['board_size'] + ' ' + processed_data['card_type']

        return processed_data
