import pandas as pd
import numpy as np

class DataAnalyzer:
    def __init__(self, data):
        self.data = data
        self.error_rates = None

    def calculate_error_rates(self):
        # 각 조합별 평균 에러율 계산
        error_rates = []
        combinations = self.data['combination'].unique()
        
        for comb in combinations:
            mask = self.data['combination'] == comb
            avg_error = self.data[mask]['error_rate'].mean()
            std_error = self.data[mask]['error_rate'].std()
            
            error_rates.append({
                'combination': comb,
                'avg_error': avg_error,
                'std_error': std_error
            })
        
        self.error_rates = pd.DataFrame(error_rates)
        self.error_rates = self.error_rates.sort_values('avg_error')
        return self.error_rates

    def get_difficulty_order(self):
        if self.error_rates is None:
            self.calculate_error_rates()
        return self.error_rates['combination'].tolist()
    
    def analyze_level_performance(self):
        # 사용자 레벨 정보 병합
        merged_data = pd.merge(
            self.data,
            self.user_levels,  # 사용자 레벨 데이터
            left_on='member_id',
            right_on='id'
        )
        
        # 레벨별, 게임 조합별 평균 error rate 계산
        level_performance = merged_data.groupby(['level', 'combination'])['error_rate'].agg(['mean', 'std']).reset_index()
        
        # 최적 난이도 찾기 (error rate 30-50% 기준)
        optimal_combinations = []
        for level in level_performance['level'].unique():
            level_data = level_performance[level_performance['level'] == level]
            optimal = level_data[
                (level_data['mean'] >= 30) & 
                (level_data['mean'] <= 50)
            ]
            if not optimal.empty:
                optimal_combinations.append({
                    'level': level,
                    'optimal_combinations': optimal['combination'].tolist(),
                    'error_rates': optimal['mean'].tolist()
                })
        
        return level_performance, optimal_combinations