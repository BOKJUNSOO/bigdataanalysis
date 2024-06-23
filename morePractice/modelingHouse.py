# predict sales prize!
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
train = pd.read_csv(r'C:\Users\brian\OneDrive\Desktop\JUNSOO\bigdata\data\house_train.csv' , encoding = 'cp949')
test = pd.read_csv(r'C:\Users\brian\OneDrive\Desktop\JUNSOO\bigdata\data\house_test.csv' , encoding = 'cp949')

# train 데이터 전처리
x_id = train['Id']
train.drop(columns = 'Id' , inplace = True)

train['Street'] = (train['Street']
                   .replace('Pave',0)
                   .replace('Grvl',1)
                   )
# MSZoning
zoning_dummy = (pd
                .get_dummies(train['MSZoning'])
                )
train = pd.concat([train , zoning_dummy] , axis = 1)
train.drop(columns = 'MSZoning' , inplace = True)

# SaleCondition
SaleC_dummy = (pd
               .get_dummies(train['SaleCondition'])
               )
train = pd.concat([train , SaleC_dummy] , axis = 1)
train.drop(columns = 'SaleCondition' , inplace = True)

train.info()

# LotFrontage
round(train['LotFrontage'].mean()  , 0)
train['LotFrontage'].fillna(70, inplace= True)

# LandContour
LandContour_dummy = (pd
               .get_dummies(train['LandContour'])
               )
train = pd.concat([train , LandContour_dummy] , axis = 1)
train.drop(columns = 'LandContour' , inplace = True)

# get_dummies for object  ()
object_dummy
train.info()



