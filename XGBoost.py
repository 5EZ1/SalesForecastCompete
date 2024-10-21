# 패키지 불러오기
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import koreanize_matplotlib
import warnings

# 경고 메시지 무시 설정
warnings.filterwarnings('ignore')

# 데이터 불러오기
def get_data():
    df = pd.read_csv('C:/공모전/유통데이터 활용 경진대회/Data/preprocessed_data.csv')
    return df

# 특성과 타겟 변수 정의 함수
def define_feature_target(df):
    X = df.drop(['판매수량', '판매일'], axis=1)
    y = df['판매수량']
    return X, y

# 데이터 분할 함수
def train_test_split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# 모델 학습 함수
def train_xgboost(X_train, y_train):
    xg_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1, max_depth=5, alpha=10, n_estimators=100)
    xg_reg.fit(X_train, y_train)
    return xg_reg

# 예측 함수
def predict_model(xg_reg, X_test):
    return xg_reg.predict(X_test)

# 모델 평가 함수
def evaluate_model(y_test, y_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse}")

# 특성 중요도 시각화 함수
def plot_feature_importance(xg_reg):
    xgb.plot_importance(xg_reg)
    plt.show()

# 미래 6개월 예측 함수
def forecast_future(xg_reg, df):
    future_dates = pd.date_range(start='2024-01-01', end='2024-06-30', freq='M')
    future_df = pd.DataFrame({'판매일': future_dates})
    future_df = future_df.merge(df.drop(columns=['판매수량', '판매일']), how='left', left_index=True, right_index=True)
    future_df.fillna(method='ffill', inplace=True)
    future_X = future_df.drop(columns=['판매일'])
    future_y_pred = xg_reg.predict(future_X)
    
    # 예측 결과 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(future_dates, future_y_pred, 'r--', label='2024년 1~6월 예측 판매수량')
    plt.title('2024년 1~6월 월별 판매수량 예측')
    plt.xlabel('날짜')
    plt.ylabel('판매수량')
    plt.legend()
    plt.grid()
    plt.show()

    # 월별 예측값 출력
    for date, value in zip(future_dates, future_y_pred):
        print(f"{date.strftime('%Y-%m')}: {value:.2f}")

# 메인 실행 함수
def main():
    # 데이터 준비
    df = get_data()
    X, y = define_feature_target(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # 모델 학습
    xg_reg = train_xgboost(X_train, y_train)

    # 예측
    y_pred = predict_model(xg_reg, X_test)

    # 평가
    evaluate_model(y_test, y_pred)

    # 미래 6개월 예측
    forecast_future(xg_reg, df)

# 실행
main()
