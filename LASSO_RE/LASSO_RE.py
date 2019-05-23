from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import math 
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, r2_score
le = LabelEncoder()
#doc file
df=pd.read_csv('House_price.csv')
#chuyen mot vai cot tu chu thanh so
y = le.fit_transform(df.iloc[:,2])
df['MSZoning'] = y
y = le.fit_transform(df.iloc[:,5])
df['Street'] = y
y = le.fit_transform(df.iloc[:,6].astype(str))
df['Alley'] = y
y = le.fit_transform(df.iloc[:,7])
df['LotShape'] = y
y = le.fit_transform(df.iloc[:,8])
df['LandContour'] = y
y = le.fit_transform(df.iloc[:,9])
df['Utilities'] = y
y = le.fit_transform(df.iloc[:,10])
df['LotConfig'] = y
y = le.fit_transform(df.iloc[:,11])
df['LandSlope'] = y
y = le.fit_transform(df.iloc[:,12])
df['Neighborhood'] = y
y = le.fit_transform(df.iloc[:,13])
df['Condition1'] = y
y = le.fit_transform(df.iloc[:,14])
df['Condition2'] = y
y = le.fit_transform(df.iloc[:,15])
df['BldgType'] = y
y = le.fit_transform(df.iloc[:,16])
df['HouseStyle'] = y
y = le.fit_transform(df.iloc[:,21])
df['RoofStyle'] = y
y = le.fit_transform(df.iloc[:,22])
df['RoofMatl'] = y
y = le.fit_transform(df.iloc[:,23])
df['Exterior1st'] = y
y = le.fit_transform(df.iloc[:,24])
df['Exterior2nd'] = y
y = le.fit_transform(df.iloc[:,25].astype(str))
df['MasVnrType'] = y
y = le.fit_transform(df.iloc[:,27].astype(str))
df['ExterQual'] = y
y = le.fit_transform(df.iloc[:,28])
df['ExterCond'] = y
y = le.fit_transform(df.iloc[:,29])
df['Foundation'] = y
y = le.fit_transform(df.iloc[:,30].astype(str))
df['BsmtQual'] = y
y = le.fit_transform(df.iloc[:,31].astype(str))
df['BsmtCond'] = y
y = le.fit_transform(df.iloc[:,32].astype(str))
df['BsmtExposure'] = y
y = le.fit_transform(df.iloc[:,33].astype(str))
df['BsmtFinType1'] = y
y = le.fit_transform(df.iloc[:,35].astype(str))
df['BsmtFinType2'] = y
y = le.fit_transform(df.iloc[:,39])
df['Heating'] = y
y = le.fit_transform(df.iloc[:,40])
df['HeatingQC'] = y
y = le.fit_transform(df.iloc[:,41])
df['CentralAir'] = y
y = le.fit_transform(df.iloc[:,42].astype(str))
df['Electrical'] = y
y = le.fit_transform(df.iloc[:,53])
df['KitchenQual'] = y
y = le.fit_transform(df.iloc[:,54])
df['Functional'] = y
y = le.fit_transform(df.iloc[:,56])
df['FireplaceQu'] = y
y = le.fit_transform(df.iloc[:,57])
df['GarageType'] = y
y = le.fit_transform(df.iloc[:,59])
df['GarageFinish'] = y
y = le.fit_transform(df.iloc[:,62])
df['GarageQual'] = y
y = le.fit_transform(df.iloc[:,63])
df['GarageCond'] = y
y = le.fit_transform(df.iloc[:,64])
df['PavedDrive'] = y
y = le.fit_transform(df.iloc[:,71])
df['PoolQC'] = y
y = le.fit_transform(df.iloc[:,72])
df['Fence'] = y
y = le.fit_transform(df.iloc[:,73])
df['MiscFeature'] = y
y = le.fit_transform(df.iloc[:,77])
df['SaleType'] = y
y = le.fit_transform(df.iloc[:,78])
df['SaleCondition'] = y

df=df.dropna()
def Lass():
    #gan du lieu x y
    x = df.iloc[:,:80]
    y =  df.iloc[:,-1]
    #chia 90% du lieu de train va 10% de test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1,random_state=2)
    #train du lieu
    #mac dinh anpha =1
    lasso = Lasso(max_iter=10e5)
    lasso.fit(x_train,y_train)
    y_pre=lasso.predict(x_test)
    print('anpha=1\nscore: ',r2_score(y_test,y_pre))
    coeff_1=np.sum(lasso.coef_!=0)
    print('tong coeff khac khong',coeff_1)
    #anpha =0.01
    lasso_001 = Lasso(alpha=0.01,max_iter=10e5)
    lasso_001.fit(x_train,y_train)
    y_pre001=lasso.predict(x_test)
    print('anpha=0.01\n score: ',r2_score(y_test,y_pre001))
    #tinh tong he so khac khong
    coeff_001=np.sum(lasso_001.coef_!=0)
    print('tong coeff khac khong',coeff_001)
    #anpha = 0.001
    lasso_001 = Lasso(alpha=0.001,max_iter=10e5)
    lasso_001.fit(x_train,y_train)
    y_pre001=lasso.predict(x_test)
    print('anpha=0.001\n score: ',r2_score(y_test,y_pre001))
    #tinh tong he so khac khong
    coeff_001=np.sum(lasso_001.coef_!=0)
    print('tong coeff khac khong',coeff_001)
    #anpha = 0.0001
    lasso_001 = Lasso(alpha=0.0001,max_iter=10e5)
    lasso_001.fit(x_train,y_train)
    y_pre001=lasso.predict(x_test)
    print('anpha=0.0001\n score: ',r2_score(y_test,y_pre001))
    #tinh tong he so khac khong
    coeff_001=np.sum(lasso_001.coef_!=0)
    print('tong coeff khac khong',coeff_001)
def Ridges():
    #gan du lieu x y
    x = df.iloc[:,:80]
    y =  df.iloc[:,-1]
    #chia 80% du lieu de train va 10% de test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1,random_state=2)
    #train du lieu
    #mac dinh anpha =1
    ride = Ridge(alpha=1)
    ride.fit(x_train,y_train)
    y_pre=ride.predict(x_test)
    print('anpha=1\nscore: ',r2_score(y_test,y_pre))
    #anpha =0.01
    ride_001 = Ridge(alpha=0.01)
    ride_001.fit(x_train,y_train)
    y_pre001=ride_001.predict(x_test)
    print('anpha=0.01\n score: ',r2_score(y_test,y_pre001))
    #anpha = 0.001
    ride_0001 = Ridge(alpha=0.001)
    ride_0001.fit(x_train,y_train)
    y_pre001=ride_0001.predict(x_test)
    print('anpha=0.001\n score: ',r2_score(y_test,y_pre001))
    #anpha = 0.0001
    ride_00001 = Ridge(alpha=0.0001)
    ride_00001.fit(x_train,y_train)
    y_pre001=ride_00001.predict(x_test)
    print('anpha=0.0001\n score: ',r2_score(y_test,y_pre001))
   
try:
    mode=int(input('Train voi dataset giá nhà\n Chon so 1 de thuc hien voi LASSO\n chon so 2 de thuc hien voi RIDGE\n Input:'))
    if mode ==1:
        Lass()
    if mode ==2:
        Ridges()
except ValueError:
    print ('Not a number')