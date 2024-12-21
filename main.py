from src.data_loader import DataLoader
from src.data_analyzer import DataAnalyzer
from src.visualizer import Visualizer
from src.linear_regression_model import train_linear_regression, plot_regression_results
from src.regression_3d import perform_3d_regression
from src.regression_3d_polynomial import perform_3d_polynomial_regression
from src.regression_polynomial import perform_polynomial_regression
import os

def main():
    # 파일 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    game_data_path = os.path.join(current_dir, 'data', '휴인미 game result 복사본.xlsx')
    user_levels_path = os.path.join(current_dir, 'data', '휴인미 member 복사본.xlsx')
    results_path = os.path.join(current_dir, 'results', 'figures')
    
    # 결과 디렉토리 생성
    os.makedirs(results_path, exist_ok=True)
    
    # 데이터 로드 및 전처리
    loader = DataLoader(game_data_path)
    data = loader.load_data()
    processed_data = loader.preprocess_data()  # combination_means는 제거

    # 데이터 분석
    analyzer = DataAnalyzer(processed_data)
    
    # 사용자 레벨 데이터 로드
    analyzer.load_user_levels(user_levels_path)
    
    # 에러율 계산
    error_rates = analyzer.calculate_error_rates()
    
    # Lilliefors 테스트 수행
    print("\nPerforming comprehensive normality tests...")
    analyzer.perform_comprehensive_normality_test(processed_data)
    
    # 레벨 기반 분석
    level_performance, optimal_combinations, _ = analyzer.analyze_level_performance()
    
    # 시각화
    visualizer = Visualizer(processed_data, error_rates)
    
    # 레벨별 성능 시각화
    visualizer.create_level_optimization_plot(level_performance)
    
    # 레벨별 추천 게임 표시
    visualizer.create_level_recommendation_plot(optimal_combinations)
    
    # 인터랙티브 체크박스 플롯 생성
    visualizer.create_interactive_plot()
    
    # ANOVA 테스트 실행
    analyzer.perform_anova_test(processed_data)
    
    # 통합 에러율 그래프 생성
    analyzer.create_combined_error_rate_plot(processed_data)
    
    # 선형 회귀 모델 학습 및 시각화 추가
    print("\nTraining Linear Regression Model...")
    model, encoder = train_linear_regression(processed_data)  # Train and return the model and encoder
    plot_regression_results(processed_data, model, encoder)  # Visualize results

    # 3D 회귀 분석 실행
    print("\nPerforming 3D Regression Analysis...")
    perform_3d_regression(processed_data)

    # Polynomial 회귀 분석 실행
    print("\nPerforming Polynomial Regression Analysis...")
    degree = 2  # 다항식 차수 설정
    perform_polynomial_regression(processed_data, degree)

    # 3D Polynomial 회귀 분석 실행
    print("\nPerforming 3D Polynomial Regression Analysis...")
    degree = 3  # 다항식 차수 설정
    perform_3d_polynomial_regression(processed_data, degree)

if __name__ == "__main__":
    main()
