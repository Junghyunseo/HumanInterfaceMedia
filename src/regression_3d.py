import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

def perform_3d_regression(data):
    """
    정렬된 데이터로 3D 회귀 분석을 수행하고 시각화
    """
    # 1. 데이터 정렬
    # preprocess_and_sort_data는 DataLoader 클래스에서 처리해야 하므로
    # 호출 전에 데이터가 이미 정렬된 상태로 전달되어야 함.

    # 2. 독립 변수와 종속 변수 정의
    X = data[['board_size', 'card_type']].copy()
    y = data['error_rate'].values

    # 3. Label Encoding
    le_board = LabelEncoder()
    le_card = LabelEncoder()
    X['board_size'] = le_board.fit_transform(X['board_size'])  # 3x3, 4x4, 5x5 -> 0, 1, 2
    X['card_type'] = le_card.fit_transform(X['card_type'])    # NUMBER, ALPHABET, etc.

    # 4. 선형 회귀 모델 학습
    model = LinearRegression()
    model.fit(X, y)

    # 5. 예측값 계산
    y_pred = model.predict(X)

    # 6. 모델 평가
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R²): {r2:.2f}")

    # 7. 3D 시각화
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 실제 데이터 산점도
    ax.scatter(X['board_size'], X['card_type'], y, color='blue', label='Actual', alpha=0.6)

    # 회귀면 생성
    X1, X2 = np.meshgrid(np.linspace(X['board_size'].min(), X['board_size'].max(), 10),
                         np.linspace(X['card_type'].min(), X['card_type'].max(), 10))
    Z = model.intercept_ + model.coef_[0] * X1 + model.coef_[1] * X2
    ax.plot_surface(X1, X2, Z, color='red', alpha=0.3, label='Regression Plane')

    # 축 레이블 설정
    ax.set_xlabel('Board Size')
    ax.set_ylabel('Card Type')
    ax.set_zlabel('Error Rate')
    ax.set_title('3D Regression Analysis (Sorted by Error Rate)')
    plt.legend()
    plt.show()

    return model
