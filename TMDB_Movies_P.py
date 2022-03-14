import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
path = os.path.dirname(os.path.abspath(__file__))
Movies =  pd.read_csv(path + "/tmdb-movies.csv")
drop = ['id','imdb_id','original_title','homepage','tagline','overview','release_date','keywords', 'director','cast','production_companies']
#Droping unsused Coloumns
Movies.drop(drop, axis = 1 , inplace = True)
#Droping Duplicates
Movies = Movies.drop_duplicates()
nan_col = ['revenue_adj','budget_adj']
Movies[nan_col] = Movies[nan_col].replace({'0':np.nan, 0:np.nan})
#removing All Null Values and Zero Values
Movies.dropna(how ='any',inplace=  True)
#making NetProfit Col
Movies['Netprofit'] = Movies.apply(lambda row: row.revenue_adj - row.budget_adj, axis=1)
drop1 = ['revenue_adj','budget_adj']
#droping unused cols.
Movies.drop(drop1,axis = 1,inplace = True)
#dummycol = pd.get_dummies(Movies["release_year"],prefix= 'year')
#Movies = pd.merge(left= Movies , right= dummycol,left_index= True,right_index= True)
# one hot encoding the year column made for overfitting makin the MSE larger than expected of 1.0E+41
gencol = Movies['genres'].str.get_dummies(sep='|')
#one Hot encoding Genres using | as separator
Movies = pd.merge(left= Movies,right= gencol,left_index= True,right_index= True)
Movies = Movies.drop(columns= 'genres')
Movies = Movies.drop(columns= 'release_year')
#Droping unused col
#puting on X all cols but netprofit
x_data =  Movies.drop(columns= 'Netprofit')
#puting on y netprofit only
Y_data = Movies[['Netprofit']]
x_train,x_test,y_train,y_test = train_test_split(x_data,Y_data,test_size= 0.30,random_state= 35)
Scalar = StandardScaler()
#scaed = MinMaxScaler()
#scaledd = pd.DataFrame(scaed.fit_transform(x_train),columns= x_data.columns )
# Minmax scaling had almost same result as the standrad scaling giving MSE of 2.4E+16
#polyfet = PolynomialFeatures(degree =3);
#scaled_train = polyfet.fit_transform(x_train)
#scaled_test =  polyfet.fit_transform(x_test)
#Regression Model
reg = LinearRegression()
#scaling X Train Data
scaled_train = pd.DataFrame(Scalar.fit_transform(x_train),columns= x_data.columns)
reg.fit(scaled_train,y_train)
#scaling X Test Data
scaled_test = pd.DataFrame(Scalar.fit_transform(x_test),columns= x_data.columns)
y_predict = reg.predict(scaled_test)
meanabserr = metrics.mean_absolute_error(y_test,y_predict)
meansqrerr = metrics.mean_squared_error(y_test,y_predict)
print('Co-efficient of linear regression',reg.coef_)
print('Intercept of linear regression model',reg.intercept_)
print("mean absolute eror = " + str(meanabserr))
print("mean square eror = " + str(meansqrerr))
print("root mean square error = " +str(np.sqrt(meansqrerr)))
print(reg.score(scaled_test,y_test))
#plt.scatter(y_test,y_predict)
#plt.show()
#x=np.arange(0,len(x_train),1)
#y=np.arange(0,len(y_train),1)
#plt.scatter(x,y)
#plt.show()
