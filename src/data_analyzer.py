import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    def create_combined_error_rate_plot(self, data):
        plt.figure(figsize=(15, 8))
        
        # 데이터 전처리 - 보드 크기와 게임 타입을 정확히 분리
        data = data.copy()  # 원본 데이터 보존을 위한 복사
        data['board_size'] = data['combination'].str.extract('(3x3|4x4|5x5)')[0]  # 정규표현식으로 추출
        data['game_type'] = data['combination'].str.extract('(NUMBER|ALPHABET|SHAPE|ARABIC)')[0]  # 정규표현식으로 추출
        
        # 박스 위치 조정을 위한 설정
        positions = {
            '3x3': [-0.3, -0.1, 0.1, 0.3],
            '4x4': [0.7, 0.9, 1.1, 1.3],
            '5x5': [1.7, 1.9, 2.1, 2.3]
        }
        
        colors = ['royalblue', 'crimson', 'forestgreen', 'darkorchid']
        game_types = ['NUMBER', 'ALPHABET', 'SHAPE', 'ARABIC']
        
        # 각 게임 타입별로 박스플롯 그리기
        for i, game_type in enumerate(game_types):
            for j, size in enumerate(['3x3', '4x4', '5x5']):
                subset = data[(data['game_type'] == game_type) & (data['board_size'] == size)]
                if not subset.empty:
                    box = plt.boxplot(subset['error_rate'], 
                                    positions=[positions[size][i]], 
                                    widths=0.15,
                                    patch_artist=True,
                                    boxprops=dict(facecolor=colors[i], alpha=0.6),
                                    medianprops=dict(color='black'),
                                    flierprops=dict(marker='o', markerfacecolor=colors[i]),
                                    showfliers=True)
        
        # 그래프 스타일링
        plt.title('Error Rates by Board Size and Game Type', fontsize=14, pad=20)
        plt.xlabel('Board Size', fontsize=12)
        plt.ylabel('Error Rate (%)', fontsize=12)
        
        # x축 설정
        plt.xticks([0, 1, 2], ['3x3', '4x4', '5x5'])
        
        # y축 범위 설정
        plt.ylim(-5, 100)
        
        # 격자 추가
        plt.grid(True, axis='y', alpha=0.3, linestyle='--')
        
        # 범례 추가
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.6) 
                        for i in range(len(game_types))]
        plt.legend(legend_elements, game_types, 
                title='Game Type', 
                title_fontsize=12, 
                fontsize=10,
                bbox_to_anchor=(1.05, 1), 
                loc='upper left')
        
        # 평균값 출력
        means = data.groupby(['board_size', 'game_type'])['error_rate'].mean()
        std = data.groupby(['board_size', 'game_type'])['error_rate'].std()
        
        print("\nMean Error Rates (%) ± Standard Deviation:")
        print("=" * 70)
        print(f"{'Board Size':<10} {'Game Type':<10} {'Mean':>8} {'Std':>6}")
        print("-" * 70)
        for size in ['3x3', '4x4', '5x5']:
            for game_type in game_types:
                if (size, game_type) in means.index:
                    mean_val = means[size, game_type]
                    std_val = std[size, game_type]
                    print(f"{size:<10} {game_type:<10} {mean_val:>8.2f} ± {std_val:>6.2f}")
        
        plt.tight_layout()
        plt.show()

    def perform_anova_test(self, data):
        """
        모든 조합에 대한 Two-way ANOVA (3x4 factorial design)
        """
        print("\nTwo-way ANOVA Test Results for All Combinations")
        print("=" * 60)
        
        # 보드 크기와 게임 타입 분리
        data['board_size'] = data['combination'].str.split().str[0]
        data['game_type'] = data['combination'].str.split().str[1]
        
        # Two-way ANOVA 수행
        from statsmodels.stats.anova import anova_lm
        from statsmodels.formula.api import ols
        
        model = ols('error_rate ~ C(board_size) + C(game_type) + C(board_size):C(game_type)', 
                    data=data).fit()
        anova_table = anova_lm(model, typ=2)
        
        print("\nANOVA Results:")
        print(anova_table)
        
        # 기술 통계량 출력
        print("\nDescriptive Statistics:")
        print("-" * 60)
        desc_stats = data.groupby(['board_size', 'game_type'])['error_rate'].agg(['mean', 'std', 'count'])
        print(desc_stats)
        
        # 시각화
        plt.figure(figsize=(15, 10))
        
        # 1. Interaction plot
        plt.subplot(2, 2, 1)
        interaction_data = data.groupby(['board_size', 'game_type'])['error_rate'].mean().unstack()
        
        for column in interaction_data.columns:
            plt.plot(interaction_data.index, interaction_data[column], 
                    marker='o', label=column, linewidth=2, markersize=8)
        
        plt.title('Interaction Effect:\nBoard Size × Game Type')
        plt.xlabel('Board Size')
        plt.ylabel('Average Error Rate (%)')
        plt.legend(title='Game Type')
        plt.grid(True, alpha=0.3)
        
        # 2. Box plots for board size
        plt.subplot(2, 2, 2)
        sns.boxplot(data=data, x='board_size', y='error_rate')
        plt.title('Error Rates by Board Size')
        
        # 3. Box plots for game type
        plt.subplot(2, 2, 3)
        sns.boxplot(data=data, x='game_type', y='error_rate')
        plt.title('Error Rates by Game Type')
        
        # 4. Interaction heatmap
        plt.subplot(2, 2, 4)
        sns.heatmap(interaction_data, annot=True, fmt='.1f', cmap='YlOrRd')
        plt.title('Error Rate Heatmap')
        
        plt.tight_layout()
        plt.show()
        
        # Post-hoc analysis
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        
        # Board size에 대한 post-hoc
        if anova_table.loc['C(board_size)', 'PR(>F)'] < 0.05:
            print("\nPost-hoc analysis for Board Size (Tukey's HSD):")
            print(pairwise_tukeyhsd(data['error_rate'], data['board_size']))
        
        # Game type에 대한 post-hoc
        if anova_table.loc['C(game_type)', 'PR(>F)'] < 0.05:
            print("\nPost-hoc analysis for Game Type (Tukey's HSD):")
            print(pairwise_tukeyhsd(data['error_rate'], data['game_type']))
        
        # Effect size 계산 (Partial Eta-squared)
        def calculate_partial_eta_squared(aov):
            aov['pes'] = aov['sum_sq'] / (aov['sum_sq'] + aov['sum_sq'].iloc[-1])
            return aov
        
        anova_with_pes = calculate_partial_eta_squared(anova_table)
        print("\nEffect Size (Partial Eta-squared):")
        print(anova_with_pes['pes'])