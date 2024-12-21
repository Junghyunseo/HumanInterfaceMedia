import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def perform_polynomial_regression(processed_data, degree=2):
    """
    다항 회귀 분석 수행
    Args:
        processed_data: 전처리된 데이터프레임
        degree: 다항식 차수 (기본값 2)
    """
    # 1. 독립 변수와 종속 변수 설정
    X = processed_data[['board_size', 'card_type']].copy()
    y = processed_data['error_rate'].values

    # 2. Label Encoding (board_size와 card_type을 숫자로 변환)
    le_board = LabelEncoder()
    le_card = LabelEncoder()
    X['board_size'] = le_board.fit_transform(X['board_size'])  # 3x3, 4x4, 5x5 -> 0, 1, 2
    X['card_type'] = le_card.fit_transform(X['card_type'])    # NUMBER, ALPHABET, etc.

    # 3. Polynomial Features 생성
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)

    # 4. 선형 회귀 모델 학습
    model = LinearRegression()
    model.fit(X_poly, y)

    # 5. 예측값 계산
    y_pred = model.predict(X_poly)

    # 6. 모델 평가
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"Polynomial Regression (Degree {degree})")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R²): {r2:.2f}")

    # 7. 시각화
    plot_polynomial_regression_results(processed_data, y, y_pred, degree, le_board, le_card)

    return model, poly

def plot_polynomial_regression_results(processed_data, y_actual, y_pred, degree, le_board, le_card):
    """
    다항 회귀 분석 결과를 시각화
    """
    plt.figure(figsize=(12, 8))

    # 실제 값과 예측 값을 비교
    plt.scatter(processed_data['combination'], y_actual, label="Actual", alpha=0.7, color="steelblue")
    plt.scatter(processed_data['combination'], y_pred, label="Predicted", alpha=0.7, color="red")

    plt.xticks(rotation=45)
    plt.title(f"Polynomial Regression (Degree {degree}): Error Rates by Combination")
    plt.xlabel("Combination")
    plt.ylabel("Error Rate (%)")
    plt.legend()
    plt.tight_layout()
    plt.show()
