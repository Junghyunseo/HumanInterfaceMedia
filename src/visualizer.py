import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import numpy as np

class Visualizer:
    def __init__(self, data, error_rates):
        self.data = data
        self.error_rates = error_rates

    def create_interactive_plot(self):
        # 그래프 영역과 체크박스 영역 설정
        fig, ax = plt.subplots(figsize=(15, 8))
        plt.subplots_adjust(right=0.8)  # 오른쪽 여백 확보
        
        lines = []
        labels = []
        
        # 각 사용자별 선 그래프 생성
        for member_id in sorted(self.data['member_id'].unique()):
            user_data = self.data[self.data['member_id'] == member_id]
            user_means = user_data.groupby('combination')['error_rate'].mean().reset_index()
            
            ordered_combinations = self.error_rates['combination'].tolist()
            user_means['combination_order'] = user_means['combination'].map(
                {comb: i for i, comb in enumerate(ordered_combinations)}
            )
            user_means = user_means.sort_values('combination_order')
            
            line, = ax.plot(user_means['combination'], 
                          user_means['error_rate'], 
                          'o-', 
                          label=f'User {member_id}',
                          visible=False)  # 초기에는 모든 선을 숨김
            
            lines.append(line)
            labels.append(f'User {member_id}')
        
        # 체크박스 생성
        rax = plt.axes([0.85, 0.3, 0.15, 0.5])  # 체크박스 위치 조정
        check = CheckButtons(rax, labels, [False] * len(labels))
        
        def func(label):
            index = labels.index(label)
            lines[index].set_visible(not lines[index].get_visible())
            plt.draw()
        
        check.on_clicked(func)
        
        # 그래프 스타일링
        plt.xticks(rotation=45)
        plt.title('Error Rates by Game Type and Size\nSelect users from the right panel')
        plt.xlabel('Game Combinations')
        plt.ylabel('Error Rate (%)')
        plt.grid(True, alpha=0.3)
        
        plt.show()

    def create_level_optimization_plot(self, level_performance):
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # 전체 평균 error rate 계산하여 난이도 순서 결정
        combination_means = level_performance.groupby('combination')['mean'].mean().sort_values()
        ordered_combinations = combination_means.index.tolist()
        
        # 레벨별로 다른 색상 사용
        levels = sorted(level_performance['level'].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(levels)))
        
        for level, color in zip(levels, colors):
            level_data = level_performance[level_performance['level'] == level]
            
            # 정렬된 순서에 맞게 데이터 재정렬
            level_data = level_data.set_index('combination').reindex(ordered_combinations).reset_index()
            
            # 에러바와 함께 플롯
            ax.errorbar(level_data['combination'], 
                    level_data['mean'],
                    yerr=level_data['std'],
                    fmt='o-',
                    label=f'Level {level}',
                    color=color,
                    alpha=0.7)
        
        # 최적 구간 표시
        ax.axhspan(30, 50, color='green', alpha=0.1, label='Optimal Range')
        
        plt.xticks(range(len(ordered_combinations)), ordered_combinations, rotation=45)
        plt.title('Error Rates by Level and Game Type\nOptimal Range (30-50%) highlighted')
        plt.xlabel('Game Combinations (ordered by difficulty)')
        plt.ylabel('Error Rate (%)')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
        
        # 난이도 순서 출력
        print("\nGame Combinations ordered by difficulty (mean error rate):")
        print("=" * 60)
        for comb, mean in combination_means.items():
            print(f"{comb:<20}: {mean:.2f}%")

    def create_level_recommendation_plot(self, optimal_combinations):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 테이블 데이터 준비
        table_data = []
        for item in optimal_combinations:
            level = item['level']
            combinations = '\n'.join(item['optimal_combinations'])
            error_rates = [f"{rate:.1f}%" for rate in item['error_rates']]
            error_rates = '\n'.join(error_rates)
            table_data.append([f"Level {level}", combinations, error_rates])
        
        # 테이블 생성
        table = ax.table(cellText=table_data,
                        colLabels=['Level', 'Recommended Games', 'Error Rates'],
                        loc='center',
                        cellLoc='left')
        
        # 테이블 스타일링
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # 셀 높이 조정
        for cell in table._cells:
            table._cells[cell].set_height(0.1)
        
        ax.axis('off')
        plt.title('Game Recommendations by Level\n(Games with Error Rates between 30-50%)')
        plt.tight_layout()
        plt.show()