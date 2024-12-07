import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.interpolate import make_interp_spline
from matplotlib.widgets import CheckButtons

class Visualizer:
    def __init__(self, data, error_rates):
        self.data = data
        self.error_rates = error_rates

    def create_boxplot(self, save_path=None):
        plt.figure(figsize=(15, 8))
        
        # 박스플롯 생성
        sns.boxplot(data=self.data, 
                   x='combination', 
                   y='error_rate',
                   order=self.error_rates['combination'])
        
        plt.xticks(rotation=45)
        plt.title('Error Rate Distribution by Game Type and Size')
        plt.xlabel('Game Combinations')
        plt.ylabel('Error Rate (%)')  # 단위를 %로 표시
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()

    def create_individual_line_plot(self, save_path=None):
        plt.figure(figsize=(15, 8))
        
        # 각 사용자별로 다른 색상의 선 그래프 생성
        for member_id in self.data['member_id'].unique():
            user_data = self.data[self.data['member_id'] == member_id]
            
            # 각 조합별 평균 error rate 계산
            user_means = user_data.groupby('combination')['error_rate'].mean().reset_index()
            
            # error_rates의 순서대로 정렬
            ordered_combinations = self.error_rates['combination'].tolist()
            user_means['combination_order'] = user_means['combination'].map(
                {comb: i for i, comb in enumerate(ordered_combinations)}
            )
            user_means = user_means.sort_values('combination_order')
            
            plt.plot(user_means['combination'], 
                    user_means['error_rate'], 
                    'o-', 
                    label=f'User {member_id}',
                    alpha=0.7)
        
        plt.xticks(rotation=45)
        plt.title('Individual Error Rates by Game Type and Size')
        plt.xlabel('Game Combinations')
        plt.ylabel('Error Rate (%)')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        plt.show()

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
        ax.set_xticklabels(ordered_combinations, rotation=45)
        ax.set_title('Error Rates by Game Type and Size\nSelect users from the right panel')
        ax.set_xlabel('Game Combinations')
        ax.set_ylabel('Error Rate (%)')
        ax.grid(True, alpha=0.3)
        
        plt.show()