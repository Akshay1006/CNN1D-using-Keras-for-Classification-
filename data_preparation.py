# Input data is of the form

#[Unique_Key,Depnt_var,X1,X2,X3....]
#Here Unique Key is the unique identifier
#Depnt_var is the dependent variable into the model taking values as 0/1
#X1,X2,X3..- All the independent Features to be utilised for Classification

import pandas as pd
import numpy as np

#There are two major data sources - Training and OOT for validation purpose

train_df=pd.read_csv('./train_data.csv')
oot_df = pd.read_csv('./oot_data.csv')

#Drop Features with High Missing Values

miss_rate=pd.DataFrame(train_df.isnull().sum()/len(train_df))
miss_rate.reset_index(inplace=True)
miss_rate.columns=['var','rate']

keep_list=list(miss_rate['var'][miss_rate.rate <= 0.7]) # We have removed features which have missing preportion greater than 70%

train_df=train_df[keep_list]
oot_df=oot_df[keep_list]

#All the missing values are imputed with 0 in this case as it was working the best. Can explore other ways like Mean,Median or k-NN based imputation

train_df.fillna(0,inplace=True)
oot_df.fillna(0,inplace=True)

train_df.reset_index(drop=True,inplace=True)
oot_df.reset_index(drop=True,inplace=True)

key_value=['unique_key','depnt_var']

ind_train=train_df.drop(key_value,axis=1)
ind_oot=oot_df.drop(key_value,axis=1)

dep_train=train_df[key_value]
dep_oot=oot_df[key_value]

#Choosing only Numerical Features for our analysis

ind_train=ind_train.select_dtypes([np.number])
ind_oot=ind_oot.select_dtypes([np.number])

#Treating Outlier values in both Train and OOT

ind_trainv1=np.minimum(ind_train,np.percentile(ind_train,95,axis=0))
ind_trainv1=np.maximum(ind_trainv1,np.percentile(ind_train,5,axis=0))

ind_ootv1=np.minimum(ind_oot,np.percentile(ind_train,95,axis=0))
ind_ootv1=np.maximum(ind_ootv1,np.percentile(ind_train,5,axis=0))

#Scaling the data Value Imputation
import sklearn
from sklearn import preprocessing

min_max_scaler=preprocessing.MinMaxScaler().fit(ind_trainv1)

train_norm=min_max_scaler.transform(ind_trainv1)
oot_norm=min_max_scaler.transform(ind_ootv1)

train_dep_data=np.array(dep_train[['depnt_var']]).astype('float32')
