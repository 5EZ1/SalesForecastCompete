# 패키지 호출
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import warnings
import koreanize_matplotlib
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler



## 데이터프레임 출력 옵션 설정
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

# UserWarning 무시
warnings.filterwarnings("ignore")

# 1) 데이터 불러오기 (CSV 파일이 없으면 엑셀 원본 데이터를 불러옴)
def read_data():
    """
    CSV 파일이 이미 존재하면 이를 불러오고,
    존재하지 않으면 엑셀 파일에서 모든 시트를 읽어 병합 후 CSV로 저장합니다.
    """
    # CSV 저장 경로 설정
    csv_path = 'C:/공모전/유통데이터 활용 경진대회/Data/중소유통물류센터 거래 데이터/df1.csv'

    # 파일이 이미 존재하는지 확인
    if os.path.exists(csv_path):
        print(f'CSV 파일을 불러옵니다: {csv_path}')
        df = pd.read_csv(csv_path)
    else:
        # 파일 경로 설정 (역슬래시를 슬래시로 변경하거나 raw string 사용)
        file_path = 'C:/공모전/유통데이터 활용 경진대회/Data/중소유통물류센터 거래 데이터/(1 데이터) 유통데이터 활용 경진대회 배포용.xlsx'

        # 모든 시트를 읽어서 하나의 데이터프레임으로 병합
        print('엑셀 원본 데이터를 불러와 병합 중...')
        sheets_dict = pd.read_excel(file_path, sheet_name=None)
        df = pd.concat(sheets_dict.values(), ignore_index=True)

        # CSV로 저장
        df.to_csv(csv_path, index=False)
        print(f'엑셀 데이터를 CSV로 저장했습니다: {csv_path}')

    return df

# # 데이터 불러오기 및 저장 실행
# if __name__ == "__main__":
#     # 데이터 불러오기 (이미 저장된 CSV 파일이 있으면 불러오고, 없으면 원본 엑셀 데이터를 불러와 CSV로 저장)
#     df = read_data()


# 2) 데이터 전처리

# '판매일' 컬럼을 datetime 형식으로 변환하고 인덱스로 설정 함수
def preprocess_date(df):
    """
    '판매일' 컬럼을 datetime 형식으로 변환하고 인덱스로 설정
    """
    df['판매일'] = pd.to_datetime(df['판매일'])

    df.set_index('판매일', inplace=True)

    return df

def reverse_preprocess_data(df):
    """
    '판매일'이 인덱스인지 확인하고, 인덱스가 맞을 경우에만 열로 변환
    """
    if df.index.name == '판매일':
        
        df = df.reset_index()

    return df

# 요일별로 판매수량 집계 (0: 월요일, 6: 일요일)
def sales_by_weekday(df):
    """
    요일별 판매수량을 집계합니다.
    """
    df['요일'] = df['판매일'].dt.dayofweek
    weekday_sales = df.groupby('요일')['판매수량'].sum()


# 불필요한 컬럼 제거 함수
def drop_columns(df):
    """
    불필요한 컬럼을 제거합니다.
    """
    # 제거할 컬럼
    drop_columns =  ['매출처코드', '옵션코드', '규격', '입수', '상품 바코드(대한상의)','상품명','대분류','소분류','구분','우편번호']

    df_drop = df.drop(drop_columns, axis=1)

    return df_drop

# 대분류 결측치 처리 함수
def fill_missing_values(df_drop):
    """
    결측치를 처리합니다.
    """
    # '대분류' 컬럼의 결측치를 '기타'로 채웁니다.
    df_drop['대분류'] = df_drop['대분류'].fillna('기타')

    return df_drop

# 결측치를 왼쪽 열의 값으로 대체하는 함수
def fill_middle_with_large(df):
    """
    중분류 컬럼의 값이 NaN일 경우 대분류 컬럼의 값으로 대체합니다.
    """
    # 중분류 컬럼이 NaN인 경우 대분류 컬럼의 값으로 대체
    df['중분류'] = df.apply(lambda row: row['대분류'] if pd.isna(row['중분류']) else row['중분류'], axis=1)
    
    return df

# 중복된 행을 제거하는 함수
def drop_duplicates(df):
    """
    중복된 행을 제거합니다.
    """
    # '판매일'이 인덱스인지 확인하고, 인덱스가 맞을 경우에만 열로 변환
    if df.index.name == '판매일':
        # 인덱스를 열로 변환
        df = df.reset_index()

    # '구분' 컬럼을 제외한 모든 컬럼을 기준으로 중복되는 그룹을 찾음
    duplicated_groups = df.duplicated(subset=['판매일', '우편번호', '매출처코드', '상품명', '판매수량'], keep=False)

    # 중복된 그룹 모두 제거
    df_cleaned = df[~duplicated_groups]

    print(f'중복 제거 후 데이터프레임 크기: {df_cleaned.shape}')

    # preprocess_date(df_cleaned)

    return df_cleaned

# '구분' 컬럼이 '반품'인 데이터를 제거하는 함수
def remove_returns(df):
    """
    '구분' 컬럼이 '반품'인 데이터를 제거합니다.
    """
    # '구분' 컬럼이 '반품'이 아닌 행만 남겨서 데이터프레임 생성
    df_filtered = df[df['구분'] != '반품']
    
    print(f"'반품' 데이터 제거 후 데이터프레임 크기: {df_filtered.shape}")
    
    return df_filtered

# 판매수량이 0 이하인 데이터를 제거하는 함수
def drop_negative_sales(df):
    """
    판매수량이 0 이하인 데이터를 제거합니다.
    """
    # '판매수량'이 0 이하인 행을 제거
    df_positive_sales = df[df['판매수량'] > 0]

    return df_positive_sales

# 판매수량이 1000 이상인 데이터를 제거하는 함수
def drop_over_sales(df):
    """
    판매수량이 1000 이상인 데이터를 제거합니다.
    """
    # '판매수량'이 1000 이상인 행을 제거
    df_positive_sales = df[df['판매수량'] < 1000]

    return df_positive_sales

# 스케일링 함수
# RobustScaler 사용
def Robust_Scaler(df, column):
    """
    '판매수량' 컬럼에 로그 변환과 RobustScaler를 적용합니다.
    """
    # 로그 변환
    df[column] = np.log1p(df[column])

    # RobustScaler 적용
    scaler = RobustScaler()
    df[column] = scaler.fit_transform(df[[column]])

    return df

# MinMaxScaler 사용
def minmax_scaling(df, columns):
    """
    지정된 열을 MinMaxScaler로 스케일링합니다.
    
    Parameters:
    df (DataFrame): 데이터프레임
    columns (list): 스케일링할 열의 이름 리스트
    
    Returns:
    DataFrame: 스케일링된 데이터프레임
    """
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

# StandardScaler 사용
def standard_scaling(df, columns):
    """
    지정된 열을 StandardScaler로 스케일링합니다.
    
    Parameters:
    df (DataFrame): 데이터프레임
    columns (list): 스케일링할 열의 이름 리스트
    
    Returns:
    DataFrame: 스케일링된 데이터프레임
    """
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

# 원핫인코딩 함수
def encoding(df,column):
    """
    원핫인코딩을 수행합니다.
    """
    # 원핫인코딩 수행
    encoded_df = pd.get_dummies(df, columns=[column])
    return encoded_df


# 3. 데이터 전처리 실행
def preprocess_df(df):
    """
    1. 결측치 처리
        1) 대분류 결측치 처리
        2) 결측치를 왼쪽 열의 값으로 대체
    2. 이상치 처리
        1) 중복 데이터 제거
        2) 구분이 '반품'인 데이터 제거
        3) '판매수량' 0 이하 데이터 제거
    3. 필요한 컬럼만 남기기
    4. '판매수량' 컬럼 로그 변환 및 RobustScaler 적용
    5. 우편번호 데이터 병합
    """

    # 대분류 결측치 처리
    df = fill_missing_values(df)
    
    # 중분류의 결측치를 대분류의 값으로 대체
    df_filled = fill_middle_with_large(df)

    # 중복 데이터 제거
    df_cleaned = drop_duplicates(df_filled)

    # 여기서 preprocess_data 함수가 실행되어 판매일은 datetime 형식으로 변환되어 있음

    # '판매수량' 0 이하 데이터 제거
    df = drop_negative_sales(df_cleaned)

    # '판매수량' 1000 이상 데이터 제거
    df = drop_over_sales(df)

    # '구분' 컬럼이 '반품'인 데이터 제거
    df = remove_returns(df)

    # 필요한 컬럼만 남기기
    df = drop_columns(df)
    
    # '중분류' 범주형 데이터를 원핫인코딩
    df = encoding(df)

    # '판매수량' 컬럼 로그 변환 및 RobustScaler 적용
    # Robust_df = Robust_Scaler(df)
    # MinMax_df = minmax_scaling(df, ['판매수량'])
    # Standard_df = standard_scaling(df, ['판매수량'])

    return df

# 데이터 전처리 실행 함수 (SARIMA용)
def preprocess_for_sarima(df):
    """
    SARIMA 모델에 맞게 데이터를 전처리합니다.
    """
    # '판매일' 컬럼을 datetime 형식으로 변환하고 인덱스로 설정합니다.
    # df = preprocess_date(df)
    # 중복 데이터 제거
    df_cleaned = drop_duplicates(df)

    # '판매수량' 0 이하 데이터 제거
    df = drop_negative_sales(df_cleaned)

    # '판매수량' 1000 이상 데이터 제거
    df = drop_over_sales(df)

    # '구분' 컬럼이 '반품'인 데이터 제거
    df = remove_returns(df)

    # 필요한 컬럼만 남기기
    df = drop_columns(df)

    # 특정 중분류만 필터링 ('라면,통조림,상온즉석')
    df = df[df['중분류'] == '라면,통조림,상온즉석']

    # 결측치 처리 (필요시 추가)
    df.dropna(subset=['판매수량'], inplace=True)

    return df

# 데이터 저장 (전처리 후의 데이터 저장 위치 및 형식 설정)
def save_csv(df, filename):
    """
    데이터프레임을 CSV 파일로 저장합니다.
    
    Parameters:
    df (DataFrame): 저장할 데이터프레임
    filename (str): 저장할 파일 이름
    """
    # DataFrame을 CSV 파일로 저장
    df.to_csv(f'C:/공모전/유통데이터 활용 경진대회/Data/{filename}.csv', index=False)

    print("CSV 파일 저장 완료!")

# # 전체 실행 함수 예시
# if __name__ == "__main__":
#     df = read_data()  # 데이터 읽기
#     df = preprocess_for_sarima(df)  # SARIMA에 맞는 전처리
#     save_csv(df, 'preprocessed_data_for_sarima')  # 전처리된 데이터 저장

# 공휴일 리스트
# 대한민국 공휴일 리스트 (2021-2023)
holidays = [
    # 2021년 공휴일
    '2021-01-01',  # 신정
    '2021-02-11', '2021-02-12', '2021-02-13',  # 설날 연휴  2021-02-11    413
    '2021-03-01',  # 삼일절  2021-03-01    579
    '2021-05-05',  # 어린이날  2021-05-05    404
    '2021-05-19',  # 부처님오신날  2021-05-19    509
    '2021-06-06',  # 현충일
    '2021-08-15',  # 광복절
    '2021-09-20', '2021-09-21', '2021-09-22',  # 추석 연휴  2021-09-20    711
    '2021-10-03',  # 개천절
    '2021-10-09',  # 한글날
    '2021-12-25',  # 성탄절
    
    # 2022년 공휴일
    '2022-01-01',  # 신정  
    '2022-01-31', '2022-02-01', '2022-02-02',  # 설날 연휴  2022-01-31    535
    '2022-03-01',  # 삼일절
    '2022-05-05',  # 어린이날
    '2022-05-08',  # 부처님오신날
    '2022-06-06',  # 현충일  2022-06-06    412
    '2022-08-15',  # 광복절
    '2022-09-09', '2022-09-10', '2022-09-11', '2022-09-12',  # 추석 연휴  2022-09-09    579
    '2022-10-03',  # 개천절
    '2022-10-09', '2022-10-10',  # 한글날 및 대체휴일
    '2022-12-25',  # 성탄절
    
    # 2023년 공휴일
    '2023-01-01',  # 신정
    '2023-01-21', '2023-01-22', '2023-01-23', '2023-01-24',  # 설날 연휴  2023-01-21    407
    '2023-03-01',  # 삼일절
    '2023-05-05',  # 어린이날
    '2023-05-27', '2023-05-29',  # 부처님오신날 및 대체휴일
    '2023-06-06',  # 현충일
    '2023-08-15',  # 광복절
    '2023-09-28', '2023-09-29', '2023-09-30', '2023-10-02',  # 추석 연휴 및 대체휴일  2023-09-28    303
    '2023-10-03',  # 개천절
    '2023-10-09',  # 한글날
    '2023-12-25',  # 성탄절
]
# 이전에는 일부 공휴일에도 판매를 했지만 최근에는 설날연휴 첫날, 추석연휴 첫날을 제외하면 대부분 휴무

df = read_data()  # 데이터 불러오기
df['판매일'] = pd.to_datetime(df['판매일'])  # '판매일' 컬럼을 datetime 형식으로 변환
# df['주말여부'] = df['판매일'].dt.weekday.isin([5, 6]).astype(int)
# df['공휴일여부'] = df['판매일'].apply(lambda x: 1 if x in holidays else 0)

# 공휴일 컬럼 추가
df['is_holiday'] = df['판매일'].isin(holidays)

# 공휴일에 해당하는 판매량만 필터링
holiday_sales = df[df['is_holiday']]


print(holiday_sales[holiday_sales['중분류'] == '라면,통조림,상온즉석'].groupby('판매일')['판매수량'].sum())
# # 공휴일별 판매량 시각화
# plt.figure(figsize=(12, 6))
# plt.bar(holiday_sales['판매일'], holiday_sales['판매수량'], color='orange', label='공휴일 판매량')
# plt.xlabel('날짜')
# plt.ylabel('판매수량')
# plt.title('2021-2023년 공휴일 판매수량 시각화')
# plt.xticks(rotation=45)
# plt.legend()
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()