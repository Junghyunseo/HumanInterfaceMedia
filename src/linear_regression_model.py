from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt


def train_linear_regression(game_data):
    """
    보드 크기와 카드 유형을 독립 변수로 하는 선형 회귀 모델 학습.
    """
    # One-Hot Encoding
    encoder = OneHotEncoder(sparse_output=False)
    encoded_features = encoder.fit_transform(game_data[['board_size', 'card_type']])
    
    # X, y 생성
    X = encoded_features
    y = game_data['error_rate'].values

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 모델 학습
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # 테스트 데이터 예측
    y_pred = lr_model.predict(X_test)

    # 음수 결과 클리핑
    y_pred = np.clip(y_pred, 0, None)

    # 모델 평가
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R²): {r2:.2f}")

    # 모델 계수 출력
    print("\nModel Coefficients:")
    feature_names = encoder.get_feature_names_out(['board_size', 'card_type'])
    for name, coef in zip(feature_names, lr_model.coef_):
        print(f"{name}: {coef:.2f}")
    print(f"Intercept: {lr_model.intercept_:.2f}")

    return lr_model, encoder

def plot_regression_results(game_data, lr_model, encoder):
    """
    보드 크기와 카드 유형 조합별로 예측 에러율과 실제 에러율 비교.
    """
    # 조합을 숫자로 매핑
    game_data['combination_numeric'] = game_data['combination'].astype('category').cat.codes
    combination_mapping = dict(enumerate(game_data['combination'].astype('category').cat.categories))

    # 예측값 계산
    encoded_features = encoder.transform(game_data[['board_size', 'card_type']])
    y_pred = lr_model.predict(encoded_features)
    y_pred = np.clip(y_pred, 0, None)  # 음수값 클리핑

    # 시각화
    plt.figure(figsize=(12, 8))
    plt.scatter(game_data['combination_numeric'], game_data['error_rate'], alpha=0.6, label='Actual', edgecolor='k')
    plt.scatter(game_data['combination_numeric'], y_pred, alpha=0.6, label='Predicted', color='red')

    # 조합 이름을 X축 레이블로 설정
    plt.xticks(ticks=list(combination_mapping.keys()), labels=list(combination_mapping.values()), rotation=45)
    plt.title("Linear Regression: Error Rates by Combination")
    plt.xlabel("Game Combination (Board Size × Card Type)")
    plt.ylabel("Error Rate (%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def plot_regression_results_with_order(processed_data, y_pred, combination_means):
    """
    지정된 x축 순서에 따라 결과를 시각화합니다.
    """
    # x축의 조합 순서를 사용자 지정 순서로 설정
    custom_order = [
        "3x3 ALPHABET", "3x3 NUMBER", "3x3 SHAPE",
        "4x4 ALPHABET", "3x3 ARABIC", "4x4 NUMBER",
        "4x4 SHAPE", "4x4 ARABIC", "5x5 NUMBER",
        "5x5 SHAPE", "5x5 ARABIC"
    ]
    
    # 조합 순서에 따라 데이터 정렬
    processed_data['combination'] = pd.Categorical(
        processed_data['combination'], categories=custom_order, ordered=True
    )
    processed_data = processed_data.sort_values('combination')

    # 실제 값과 예측 값을 시각화
    plt.figure(figsize=(12, 8))
    plt.scatter(
        processed_data['combination'],
        processed_data['error_rate'],
        label="Actual",
        alpha=0.7,
        color="steelblue"
    )
    plt.scatter(
        processed_data['combination'],
        y_pred,
        label="Predicted",
        alpha=0.7,
        color="red"
    )
    
    plt.xticks(rotation=45)
    plt.title("Linear Regression: Error Rates by Combination (Custom Order)")
    plt.xlabel("Combination")
    plt.ylabel("Error Rate (%)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # R² 값 출력
    print("\nCustom Order Combination Means:")
    print(combination_means.loc[custom_order].dropna())  # 지정된 순서로 정렬된 조합별 평균 에러율 출력