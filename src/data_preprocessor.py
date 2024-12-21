import pandas as pd

def preprocess_data(self):
    if self.data is None:
        self.load_data()
    
    # 1. 필요한 열 복사
    processed_data = self.data[['member_id', 'card_count', 'card_type', 'wrong_count']].copy()
    
    # 2. 누락된 값 제거 및 유효성 검사
    processed_data = processed_data.dropna()  # 누락된 값 제거
    processed_data = processed_data[processed_data['card_count'] > 0]  # 유효한 card_count만 남김
    processed_data = processed_data[processed_data['wrong_count'] >= 0]  # 음수가 아닌 wrong_count만 남김

    # 3. 보드 크기 변환
    processed_data['board_size'] = processed_data['card_count'].map({
        9: '3x3',
        16: '4x4',
        25: '5x5'
    })

    # 4. 에러율 계산
    processed_data['error_rate'] = (processed_data['wrong_count'] / processed_data['card_count']) * 100

    # 5. 에러율이 0인 데이터도 포함
    # (필터링을 하지 않음, 필요한 경우 주석 해제)
    # processed_data = processed_data[processed_data['error_rate'] > 0]

    # 6. 조합 열 생성
    processed_data['combination'] = processed_data['board_size'] + ' ' + processed_data['card_type']

    # 7. combination_means 계산 (combination별 평균 에러율)
    combination_means = processed_data.groupby('combination')['error_rate'].mean().sort_values()

    # 8. 데이터와 combination_means 반환
    return processed_data, combination_means
