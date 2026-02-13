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
transformer = ColumnTransformer(transformers=[
                  ('tnf1',OrdinalEncoder(categories=[['male','female']]),['sex']),
                  ('tnf2',OrdinalEncoder(categories=[['no','yes']]),['smoker']),
                  ('tnf3',OneHotEncoder(sparse_output=False, drop='first'),['region'])],
                  remainder='passthrough')

print(transformer.fit_transform(x_train).shape)
print(transformer.transform(x_test).shape)