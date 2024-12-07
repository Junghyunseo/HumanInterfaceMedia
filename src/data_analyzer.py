import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

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
    
    def perform_comprehensive_normality_test(self, data):
        """
        네 가지 수준에서 정규성 검정 수행
        1. 개별 조합별 (12개)
        2. 보드 크기별 (3개)
        3. 게임 종류별 (4개)
        4. 전체 통합
        """
        # 1. 개별 조합 분석
        print("\n1. Lilliefors Test Results for Each Combination:")
        print("=" * 60)
        print(f"{'Combination':<20} {'Statistic':<10} {'p-value':<10} {'Normal?':<10}")
        print("-" * 60)
        
        for combination in sorted(data['combination'].unique()):
            subset = data[data['combination'] == combination]['error_rate']
            if len(subset) > 3:
                z_scores = (subset - np.mean(subset)) / np.std(subset)
                statistic, p_value = stats.kstest(z_scores, 'norm')
                is_normal = "Yes" if p_value > 0.05 else "No"
                print(f"{combination:<20} {statistic:.4f}    {p_value:.4f}    {is_normal}")
        print("=" * 60)

        # 2. 보드 크기별 분석
        print("\n2. Lilliefors Test Results by Board Size:")
        print("=" * 60)
        print(f"{'Board Size':<20} {'Statistic':<10} {'p-value':<10} {'Normal?':<10}")
        print("-" * 60)
        
        for size in ['3x3', '4x4', '5x5']:
            size_data = data[data['combination'].str.startswith(size)]['error_rate']
            if len(size_data) > 3:
                z_scores = (size_data - np.mean(size_data)) / np.std(size_data)
                statistic, p_value = stats.kstest(z_scores, 'norm')
                is_normal = "Yes" if p_value > 0.05 else "No"
                print(f"{size:<20} {statistic:.4f}    {p_value:.4f}    {is_normal}")
        print("=" * 60)

        # 3. 게임 종류별 분석
        print("\n3. Lilliefors Test Results by Game Type:")
        print("=" * 60)
        print(f"{'Game Type':<20} {'Statistic':<10} {'p-value':<10} {'Normal?':<10}")
        print("-" * 60)
        
        for game_type in ['NUMBER', 'ALPHABET', 'SHAPE', 'ARABIC']:
            type_data = data[data['card_type'] == game_type]['error_rate']
            if len(type_data) > 3:
                z_scores = (type_data - np.mean(type_data)) / np.std(type_data)
                statistic, p_value = stats.kstest(z_scores, 'norm')
                is_normal = "Yes" if p_value > 0.05 else "No"
                print(f"{game_type:<20} {statistic:.4f}    {p_value:.4f}    {is_normal}")
        print("=" * 60)

        # 4. 전체 통합 분석
        print("\n4. Lilliefors Test Result for Overall Distribution:")
        print("=" * 60)
        print(f"{'Analysis Type':<20} {'Statistic':<10} {'p-value':<10} {'Normal?':<10}")
        print("-" * 60)
        
        all_data = data['error_rate']
        z_scores = (all_data - np.mean(all_data)) / np.std(all_data)
        statistic, p_value = stats.kstest(z_scores, 'norm')
        is_normal = "Yes" if p_value > 0.05 else "No"
        print(f"{'Overall':<20} {statistic:.4f}    {p_value:.4f}    {is_normal}")
        print("=" * 60)
        print("\nNote: p-value > 0.05 indicates normal distribution")

        # QQ plots for all four analyses
        plt.figure(figsize=(15, 10))

        # 1. Overall QQ Plot
        plt.subplot(2, 2, 1)
        stats.probplot(all_data, dist="norm", plot=plt)
        plt.title("Q-Q Plot: Overall")

        # 2. Board Size QQ Plots
        plt.subplot(2, 2, 2)
        size_colors = {'3x3': 'red', '4x4': 'blue', '5x5': 'green'}
        for size, color in size_colors.items():
            size_data = data[data['combination'].str.startswith(size)]['error_rate']
            # 이론적 분위수와 샘플 분위수 계산
            probplot_output = stats.probplot(size_data, dist="norm", fit=False)
            theoretical_quantiles = probplot_output[0]
            sample_quantiles = probplot_output[1]
            plt.plot(theoretical_quantiles, sample_quantiles, 
                    marker='o', color=color, linestyle='', label=size)
        plt.title("Q-Q Plot: Board Sizes")
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Sample Quantiles")
        plt.legend()

        # 3. Game Type QQ Plots
        plt.subplot(2, 2, 3)
        type_colors = {
            'NUMBER': 'blue',
            'ALPHABET': 'red',
            'SHAPE': 'green',
            'ARABIC': 'purple'
        }
        type_markers = {
            'NUMBER': 'o',
            'ALPHABET': 's',
            'SHAPE': '^',
            'ARABIC': 'D'
        }
        for game_type in type_colors.keys():
            type_data = data[data['card_type'] == game_type]['error_rate']
            # 이론적 분위수와 샘플 분위수 계산
            probplot_output = stats.probplot(type_data, dist="norm", fit=False)
            theoretical_quantiles = probplot_output[0]
            sample_quantiles = probplot_output[1]
            plt.plot(theoretical_quantiles, sample_quantiles,
                    marker=type_markers[game_type], 
                    color=type_colors[game_type], 
                    linestyle='', 
                    label=game_type)
        plt.title("Q-Q Plot: Game Types")
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Sample Quantiles")
        plt.legend()

        # 4. Individual Combinations (showing trend)
        plt.subplot(2, 2, 4)
        comb_means = data.groupby('combination')['error_rate'].mean()
        stats.probplot(comb_means, dist="norm", plot=plt)
        plt.title("Q-Q Plot: Combination Means")

        plt.tight_layout()
        plt.show()

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