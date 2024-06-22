import pandas as pd
x_test = pd.read_csv(r'C:\Users\brian\OneDrive\Desktop\JUNSOO\bigdata\data\bike_x_test.csv' , encoding = 'cp949')
x_train = pd.read_csv(r'C:\Users\brian\OneDrive\Desktop\JUNSOO\bigdata\data\bike_x_train.csv' , encoding = 'cp949')
y_train = pd.read_csv(r'C:\Users\brian\OneDrive\Desktop\JUNSOO\bigdata\data\bike_y_train.csv' , encoding = 'cp949')


# train data 확인하기
data = pd.concat([x_train , y_train] , axis = 1)
print(data.head())

# 컬럼간 종속변수에 영향 끼치는지 EDA
# 갯수 예측 -> 합이 중요? // 생존or 명목형 여부 -> 합으로 확인 or 1에 가까운가
# groupby() -> 기준되는 컬럼으로 indexing**
print(data.groupby(['공휴일'])['count'].sum()) #공휴일 : 포함                                           
print(data.groupby(['계절'])['count'].sum())    # 계절 : 미포함

data.drop(columns = '계절' , inplace = True)    #계속된 분석을 수월하게 하기위해 data 도 drop
x_train.drop(columns = '계절' , inplace = True)
x_test.drop(columns= '계절' , inplace = True)

print(data.groupby(['날씨'])['count'].sum())    # 날씨 : 포함

print(data.groupby(['근무일'])['count'].sum())     # 포함

x_datatime = x_train['datetime']          # 제출용 저장

y_train.drop(columns = '癤풼atetime' , inplace = True)
data.drop(columns = '癤풼atetime' , inplace = True)

data.drop(columns = '체감온도' , inplace = True)    # 체감온도와 온도 상관관계가 높음
x_train.drop(columns = '체감온도' , inplace = True)
x_test.drop(columns = '체감온도' , inplace = True)
x_train['온도'] = round(x_train['온도'] , 0)


# datetime 처리
x_train['datetime'] = pd.to_datetime(x_train['datetime'])   # datetime columns를 'datetime 데이터 타입으로 바꾼다'
x_train['hour'] = x_train['datetime'].dt.hour
x_train['year'] = x_train['datetime'].dt.year
x_train['month'] = x_train['datetime'].dt.month
x_train['dayofweek'] = x_train['datetime'].dt.dayofweek

data2 = pd.concat([x_train , y_train] , axis = 1)
data2.head()

data2.groupby(['hour'])['count'].sum()      # hour : 포함
data2.groupby(['year'])['count'].sum()      # year : 포함
data2.groupby(['month'])['count'].sum()     # month : drop
data2.groupby(['dayofweek'])['count'].sum() # dayofweek : drop

x_train.drop(columns = ['datetime' , 'month' , 'dayofweek'] , inplace = True)
x_train.head()
x_test.head()
# test data도 똑같이 컬럼생성
x_test['datetime'] = pd.to_datetime(x_test['datetime'])
x_test['hour'] = x_test['datetime'].dt.hour
x_test['year'] = x_test['datetime'].dt.year
x_test.drop(columns = 'datetime', inplace = True)       #x_train , x_test 전처리 완료
x_test.head()
x_train.head()

# train data 나누기
import sklearn
from sklearn.model_selection import train_test_split

X_TRAIN , X_TEST , Y_TRAIN, Y_TEST = train_test_split(x_train, y_train , test_size = 0.2)

from xgboost import XGBRegressor            # 명목형 분류가 아닌 어떤 count 의 추측 모델 !

# 학습시키기
model = XGBRegressor(n_estimators = 100 , max_depth = 3 , random_state = 10)      # 객체 생성시에 파라미터 설정
model.fit(X_TRAIN, Y_TRAIN)

# 예측하기 / 처음 주어진 x_test 데이터 이용
y_test_predicted = pd.DataFrame(model.predict(x_test)).rename(columns = {0:'count'})

y_test_predicted[y_test_predicted['count'] < 0] = 0 

# 평가하기
Y_TEST_PREDICTED = pd.DataFrame(model.predict(X_TEST))
Y_TEST_PREDICTED[Y_TEST_PREDICTED[0] < 0 ] = 0

from sklearn.metrics import r2_score
print(r2_score(Y_TEST_PREDICTED , Y_TEST))



