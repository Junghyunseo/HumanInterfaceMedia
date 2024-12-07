from src.data_loader import DataLoader
from src.data_analyzer import DataAnalyzer
from src.visualizer import Visualizer
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
    processed_data = loader.preprocess_data()
    
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

if __name__ == "__main__":
    main()