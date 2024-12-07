from src.data_loader import DataLoader
from src.data_analyzer import DataAnalyzer
from src.visualizer import Visualizer
import os

def main():
    # 파일 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'data', '휴인미 game result 복사본.xlsx')
    results_path = os.path.join(current_dir, 'results', 'figures')
    
    # 결과 디렉토리 생성
    os.makedirs(results_path, exist_ok=True)
    
    # 데이터 로드 및 전처리
    loader = DataLoader(data_path)
    data = loader.load_data()
    processed_data = loader.preprocess_data()
    
    # 데이터 분석
    analyzer = DataAnalyzer(processed_data)
    error_rates = analyzer.calculate_error_rates()
    
    # 시각화
    visualizer = Visualizer(processed_data, error_rates)
    
    # 박스플롯 생성 및 저장
    boxplot_path = os.path.join(results_path, 'boxplot.png')
    visualizer.create_boxplot(boxplot_path)
    
    # 개별 사용자 라인 플롯 생성 및 저장
    line_plot_path = os.path.join(results_path, 'individual_lines.png')
    visualizer.create_individual_line_plot(line_plot_path)

    # 인터랙티브 체크박스 플롯 생성
    visualizer.create_interactive_plot()

if __name__ == "__main__":
    main()