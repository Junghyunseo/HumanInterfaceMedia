import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def perform_3d_polynomial_regression(processed_data, degree=2):
    """
    3D 다항 회귀 분석 수행 및 시각화
    Args:
        processed_data: 전처리된 데이터프레임
        degree: 다항식 차수 (기본값 2)
    """
    # 1. 독립 변수(X)와 종속 변수(y) 설정
    X = processed_data[['board_size', 'card_type']].copy()
    y = processed_data['error_rate'].values

    # 2. 수동 매핑으로 board_size와 card_type 변환
    board_size_mapping = {"3x3": 0, "4x4": 1, "5x5": 2}
    card_type_mapping = {"ALPHABET": 0, "SHAPE": 1, "NUMBER": 2, "ARABIC": 3}
    X['board_size'] = X['board_size'].map(board_size_mapping)
    X['card_type'] = X['card_type'].map(card_type_mapping)

    # 3. Polynomial Features 생성
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)

    # 4. 모델 학습
    model = LinearRegression()
    model.fit(X_poly, y)

    # 5. 예측값 계산
    y_pred = model.predict(X_poly)

    # 6. 모델 평가
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"3D Polynomial Regression (Degree {degree})")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R²): {r2:.2f}")

    # 7. 3D 시각화
    plot_3d_polynomial_regression(X, y, y_pred, degree, model, poly)

    return model, poly

def plot_3d_polynomial_regression(X, y, y_pred, degree, model, poly):
    """
    3D 다항 회귀 분석 결과를 시각화
    """
    # 3D 그래프 설정
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 실제 데이터 산점도
    ax.scatter(X['board_size'], X['card_type'], y, color='blue', label='Actual', alpha=0.6)

    # 예측 데이터 산점도
    ax.scatter(X['board_size'], X['card_type'], y_pred, color='red', label='Predicted', alpha=0.6)

    # 회귀 평면 생성
    X1, X2 = np.meshgrid(
        np.linspace(X['board_size'].min(), X['board_size'].max(), 20),
        np.linspace(X['card_type'].min(), X['card_type'].max(), 20)
    )
    X_grid = np.c_[X1.ravel(), X2.ravel()]
    X_grid_poly = poly.transform(X_grid)  # Polynomial 변환
    Z = model.predict(X_grid_poly).reshape(X1.shape)

    ax.plot_surface(X1, X2, Z, color='green', alpha=0.3, label='Regression Plane')

    # 축 설정
    ax.set_xlabel('Board Size')
    ax.set_ylabel('Card Type')
    ax.set_zlabel('Error Rate')
    ax.set_title(f"3D Polynomial Regression (Degree {degree})")
    ax.legend()
    plt.show()
