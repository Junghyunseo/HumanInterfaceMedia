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
        
        # 데이터 복사본 생성
        processed_data = self.data[['member_id', 'card_count', 'card_type', 'wrong_count', 'solved_time']].copy()
        
        # 보드 크기 변환
        processed_data.loc[:, 'board_size'] = processed_data['card_count'].map({
            9: '3x3',
            16: '4x4',
            25: '5x5'
        })
        
        # error rate 계산
        processed_data.loc[:, 'error_rate'] = (processed_data['wrong_count'] / processed_data['card_count']) * 100
        
        # 조합 컬럼 생성
        processed_data.loc[:, 'combination'] = processed_data['board_size'] + ' ' + processed_data['card_type']
        
        return processed_data

if __name__ == "__main__":
    loader = DataLoader("data/휴인미 game result 복사본.xlsx")
    data = loader.load_data()
    processed_data = loader.preprocess_data()
    print("\nProcessed data:")
    print(processed_data.head())