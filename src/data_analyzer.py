import pandas as pd
import numpy as np
from scipy import stats

class DataAnalyzer:
    def __init__(self, data):
        self.data = data
        self.error_rates = None
        self.user_levels = None

    def load_user_levels(self, file_path):
        """사용자 레벨 데이터 로드"""
        try:
            self.user_levels = pd.read_excel(file_path)
            print("User levels loaded successfully!")
            return self.user_levels
        except Exception as e:
            print(f"Error loading user levels: {e}")
            return None

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
    
    def perform_lilliefors_test(self, data):
        """
        각 게임 조합별로 정규성 검정 수행 (Kolmogorov-Smirnov test)
        """
        print("\nLilliefors Test Results:")
        print("=" * 60)
        print(f"{'Combination':<20} {'Statistic':<10} {'p-value':<10} {'Normal?':<10}")
        print("-" * 60)
        
        for combination in sorted(data['combination'].unique()):
            subset = data[data['combination'] == combination]['error_rate']
            if len(subset) > 3:  # 데이터가 충분한 경우에만 검정
                # 데이터를 표준화
                z_scores = (subset - np.mean(subset)) / np.std(subset)
                # Kolmogorov-Smirnov test with normal distribution
                statistic, p_value = stats.kstest(z_scores, 'norm')
                is_normal = "Yes" if p_value > 0.05 else "No"
                print(f"{combination:<20} {statistic:.4f}    {p_value:.4f}    {is_normal}")
        
        print("=" * 60)
        print("Note: p-value > 0.05 indicates normal distribution")
    
    def analyze_level_performance(self):
        if self.user_levels is None:
            raise ValueError("User levels not loaded. Please call load_user_levels() first.")
        
        # 데이터 병합 및 처리
        user_levels_slim = self.user_levels[['id', 'level']]
        merged_data = pd.merge(
            self.data,
            user_levels_slim,
            left_on='member_id',
            right_on='id',
            how='left'
        )
        
        # 레벨별, 게임 조합별 평균 error rate 계산
        level_performance = merged_data.groupby(['level', 'combination'])['error_rate'].agg(['mean', 'std']).reset_index()
        
        # 최적 난이도 찾기 (error rate 30-50% 기준)
        optimal_combinations = []
        for level in sorted(level_performance['level'].unique()):
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
        
        return level_performance, optimal_combinations, None