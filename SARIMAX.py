# 패키지 불러오기
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
import koreanize_matplotlib
import Source_Code.data1_preprocessing as data1_preprocessing

# 경고 메시지 무시 설정
warnings.filterwarnings('ignore')

# 데이터 불러오기
def get_data():
    # 데이터 읽기
    df = pd.read_csv('C:/공모전/유통데이터 활용 경진대회/Data/preprocessed_data_for_sarima.csv')
    return df

# 특정 중분류 데이터 필터링 함수 ('라면,통조림,상온즉석'에 해당하는 데이터만 사용)
def filter_data(df):
    filtered_df = df[df['중분류'] == '라면,통조림,상온즉석']
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
def train_sarimax(ts, exog):
    # SARIMA 모델 정의
    p, d, q = 1, 1, 1  # ARIMA 부분 파라미터
    P, D, Q, s = 1, 1, 1, 12  # 계절성 파라미터 (계절 주기 12개월)

    sarima_model = sm.tsa.statespace.SARIMAX(ts,
                                             order=(p, d, q),
                                             seasonal_order=(P, D, Q, s),
                                             exog=exog,
                                             enforce_stationarity=False,
                                             enforce_invertibility=False)

    # 모델 학습
    sarima_result = sarima_model.fit(disp=False)

    # 모델 요약 출력
    print(sarima_result.summary())

    return sarima_result

# 미래 6개월 예측 함수
def forecast_future(sarimax_result, ts, exog_future):
    future_steps = 6
    forecast = sarimax_result.get_forecast(steps=future_steps, exog=exog_future)
    forecast_mean = forecast.predicted_mean
    confidence_intervals = forecast.conf_int()

    # 예측 결과 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(ts.index, ts, label='실제 월별 판매수량', color='blue')
    plt.plot(forecast_mean.index, forecast_mean, 'r--', label='예측 월별 판매수량')
    plt.fill_between(forecast_mean.index,
                     confidence_intervals.iloc[:, 0],
                     confidence_intervals.iloc[:, 1],
                     color='pink', alpha=0.3)
    plt.title('라면,통조림,상온즉석 월별 판매수량 예측')
    plt.xlabel('날짜')
    plt.ylabel('판매수량')
    plt.legend()
    plt.grid()
    plt.show()

    # 월별 예측값 출력
    for date, value in zip(forecast_mean.index, forecast_mean):
        print(f"{date.strftime('%Y-%m')}: {value:.0f}")

# 공휴일과 주말 고려하여 SARIMAX 모델 적용 준비 함수
def adjust_for_holidays_and_weekends(df):
    # 공휴일 리스트 정의 (2021-2023년)
    holidays = [
        '2021-01-01', '2021-02-11', '2021-02-12', '2021-02-13', '2021-03-01',
        '2021-05-05', '2021-05-19', '2021-06-06', '2021-08-15', '2021-09-20',
        '2021-09-21', '2021-09-22', '2021-10-03', '2021-10-09', '2021-12-25',
        '2022-01-01', '2022-01-31', '2022-02-01', '2022-02-02', '2022-03-01',
        '2022-05-05', '2022-05-08', '2022-06-06', '2022-08-15', '2022-09-09',
        '2022-09-10', '2022-09-11', '2022-09-12', '2022-10-03', '2022-10-09',
        '2022-10-10', '2022-12-25', '2023-01-01', '2023-01-21', '2023-01-22',
        '2023-01-23', '2023-01-24', '2023-03-01', '2023-05-05', '2023-05-27',
        '2023-05-29', '2023-06-06', '2023-08-15', '2023-09-28', '2023-09-29',
        '2023-09-30', '2023-10-02', '2023-10-03', '2023-10-09', '2023-12-25'
    ]

    # '판매일' 컬럼을 datetime 형식으로 변환
    df['판매일'] = pd.to_datetime(df['판매일'])

    # 공휴일 컬럼 추가
    df['is_holiday'] = df['판매일'].isin(holidays).astype(int)

    # 요일 컬럼 추가 (0: 월요일, 6: 일요일)
    df['day_of_week'] = df['판매일'].dt.dayofweek

    # 주말 여부 컬럼 추가 (토요일, 일요일을 주말로 설정)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # 공휴일과 주말을 고려하여 판매량에 영향을 줄 수 있는 특성 추가
    exog = df[['판매일', 'is_holiday', 'is_weekend']].set_index('판매일')

    # 월별로 리샘플링
    exog_resampled = exog.resample('M').sum()

    return df, exog_resampled

# 메인 실행 함수
def main():
    # 데이터 준비
    df = get_data()
    df, exog = adjust_for_holidays_and_weekends(df)
    filtered_df = filter_data(df)
    ts = prepare_time_series(filtered_df)

    # 필터링된 시계열 인덱스에 맞게 외생 변수도 필터링
    exog_filtered = exog.loc[ts.index]

    # SARIMAX 모델 학습
    sarimax_result = train_sarimax(ts, exog_filtered)

    # 미래 6개월 예측을 위한 외생 변수 생성
    future_dates = pd.date_range(start=ts.index[-1] + pd.offsets.MonthBegin(), periods=6, freq='M')
    future_exog = pd.DataFrame({
        'is_holiday': [0] * 6,  # 미래 공휴일 여부는 0으로 가정
        'is_weekend': [int(date.weekday() in [5, 6]) for date in future_dates]
    }, index=future_dates)

    # 미래 6개월 예측
    forecast_future(sarimax_result, ts, future_exog)

# 실행
main()
