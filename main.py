import pandas  as pd
import numpy as np

df=pd.read_csv("C:\\Users\\jayus\\OneDrive\\Desktop\\ml\\insurance - insurance.csv")
print(df.head(3))

print(df.isnull().sum())
x=df.drop(columns=['charges'])
y=df.drop(columns=['charges'])
y=df['charges']

from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42)
print(df.shape)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


ohe = OneHotEncoder(drop='first', sparse_output=False)

x_train_sex_smoker_region = ohe.fit_transform(x_train[['sex','smoker','region']])
x_test_sex_smoker_region = ohe.transform(x_test[['sex','smoker','region']])



x_train_age_bmi_children = x_train.drop(columns =['smoker', 'region','sex']).values
x_test_age_bmi_children = x_test.drop(columns =['smoker', 'region','sex']).values
print(x_train_age_bmi_children.shape)


x_train_transformed = np.concatenate((x_train_age_bmi_children ,x_train_sex_smoker_region) , axis = 1)
print(x_train_transformed.shape)



transformers = ColumnTransformer(transformers=[
    ('tnf1', OneHotEncoder(drop='first', sparse_output=False), ['sex','smoker','region'])
], remainder='passthrough')

x_train_transformed = transformers.fit_transform(x_train)
x_test_transformed = transformers.transform(x_test)

print(x_train_transformed.shape)
print(x_test_transformed.shape)
