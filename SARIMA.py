# 패키지 불러오기
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
import koreanize_matplotlib

# 경고 메시지 무시 설정
warnings.filterwarnings('ignore')

# 데이터 불러오기
def get_data():
    # 데이터 읽기
    df1 = pd.read_csv('C:/공모전/유통데이터 활용 경진대회/Data/preprocessed_data_for_sarima.csv')
    df2 = pd.read_csv('C:/공모전/유통데이터 활용 경진대회/Data/preprocessed_data2_for_sarima.csv')
    return df1, df2

# 특정 중분류 데이터 필터링 함수 ('라면,통조림,상온즉석'에 해당하는 데이터만 사용)
def filter_data1(df):
    filtered_df = df[df['중분류'] == '라면,통조림,상온즉석']
    return filtered_df

def filter_data2(df):  
    filtered_df = df[df['대분류'] == '면류.라면류']
    return filtered_df

# 시계열 데이터 준비 함수
def prepare_time_series(df):
    # '판매일' 컬럼을 datetime 형식으로 변환하고 인덱스로 설정
    df['판매일'] = pd.to_datetime(df['판매일'])
    df.set_index('판매일', inplace=True)

    # 월별로 '판매수량'을 집계하여 시계열 데이터 준비
    ts = df['판매수량'].resample('M').sum()
    return ts

# SARIMA 모델 학습 함수
def train_sarima(ts):
    # SARIMA 모델 정의
    p, d, q = 1, 1, 1  # ARIMA 부분 파라미터
    P, D, Q, s = 1, 1, 1, 12  # 계절성 파라미터 (계절 주기 12개월)

    sarima_model = sm.tsa.statespace.SARIMAX(ts,
                                             order=(p, d, q),
                                             seasonal_order=(P, D, Q, s),
                                             enforce_stationarity=False,
                                             enforce_invertibility=False)

    # 모델 학습
    sarima_result = sarima_model.fit(disp=False)

    # 모델 요약 출력
    print(sarima_result.summary())

    return sarima_result

# 미래 6개월 예측 함수
def forecast_future(sarima_result, ts):
    future_steps = 6  # 6개월 예측
    forecast = sarima_result.get_forecast(steps=future_steps)

    # 예측된 값과 신뢰 구간
    forecast_mean = forecast.predicted_mean
    confidence_intervals = forecast.conf_int()

    # 예측 결과 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(ts.index, ts, label='실제 월별 판매수량', color='blue')
    plt.plot(forecast_mean.index, forecast_mean, 'r--', label='예측 월별 판매수량', color='red')
    plt.fill_between(forecast_mean.index,
                     confidence_intervals.iloc[:, 0],
                     confidence_intervals.iloc[:, 1],
                     color='pink', alpha=0.3)
    plt.title('Target 월별 판매수량 예측')
    plt.xlabel('날짜')
    plt.ylabel('판매수량')
    plt.legend()
    plt.grid()
    plt.show()

    # 월별 예측값 출력
    for date, value in zip(forecast_mean.index, forecast_mean):
        print(f"{date.strftime('%Y-%m')}: {value:.0f}")


# 메인 실행 함수
def main_data1():
    # 데이터 준비
    df1, _ = get_data()


    filtered_df = filter_data1(df1)
    ts = prepare_time_series(filtered_df)

    # SARIMA 모델 학습
    sarima_result = train_sarima(ts)

    # 미래 6개월 예측
    forecast_future(sarima_result, ts)

def main_data2():
    # 데이터 준비
    _, df2 = get_data()
    df2.head()
    
    filtered_df = filter_data2(df2)
    ts = prepare_time_series(filtered_df)

    # SARIMA 모델 학습
    sarima_result = train_sarima(ts)

    # 미래 6개월 예측
    forecast_future(sarima_result, ts)

# 실행
if __name__ == "__main__":
    # main_data1()
    main_data2()

