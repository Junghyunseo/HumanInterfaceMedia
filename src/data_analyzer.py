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