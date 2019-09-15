import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
#importing the datasets
cab_rides = pd.read_csv(r"C:\Users\S533718\Desktop\Datasets\uber-lyft-cab-prices\cab_rides.csv")
weather = pd.read_csv(r"C:\Users\S533718\Desktop\Datasets\uber-lyft-cab-prices\weather.csv")

# cleaning the datasets
cab_rides.isnull().sum()
cab_rides.dtypes
x_cab=cab_rides.dropna()
x_cab.describe()
x_cab.isnull().sum()
weather.isnull().sum()
y_weather=weather.fillna(weather.mean())
y_weather.isnull().sum()
y_weather.describe()
#adding the new coloumn date and time based on time stamp
x_cab["time_stamp"]
y_weather["time_stamp"]
x_cab['date_time'] = pd.to_datetime(x_cab['time_stamp']/1000, unit="s")
x_cab.head()
y_weather['date_time'] = pd.to_datetime(y_weather['time_stamp'],unit="s")
y_weather.head()
# Converting the dataetypes and merging 
x_cab['date_source_merge'] = x_cab.source.astype(str) +" - "+ x_cab.date_time.dt.date.astype("str") +" - "+ x_cab.date_time.dt.hour.astype("str")
y_weather['date_source_merge']=y_weather.location.astype(str) +" - "+ y_weather.date_time.dt.date.astype("str") +" - "+y_weather.date_time.dt.hour.astype("str")

y_weather.index = y_weather["date_source_merge"]
x_cab.dtypes
y_weather.dtypes
cab_weather = pd.merge(x_cab,y_weather,on='date_source_merge')
cab_weather.info()
cab_weather.describe()
cab_weather.isnull().sum().count
cab_weather['rain'].fillna(0,inplace=True)
cab_weather = cab_weather[pd.notnull(cab_weather['date_time_y'])]
cab_weather = cab_weather[pd.notnull(cab_weather['price'])]
cab_weather['day'] = cab_weather.date_time_x.dt.dayofweek
cab_weather['day'].describe()
cab_weather['hour'] = cab_weather.date_time_x.dt.hour
cab_weather['day'].describe()
cab_weather.columns
cab_weather.count()

#dividing the features into independant and dependent variables for lyft
X = cab_weather[cab_weather.product_id=='lyft'][['day','distance','hour','temp','clouds', 'pressure','humidity', 'wind', 'rain']]
Y = cab_weather[cab_weather.product_id=='lyft'][['price']]
X.reset_index(inplace=True)
X = X.drop(columns=['index'])
#feature scaling and converting the labels and features into arrays
features = pd.get_dummies(X)
features.columns
labels = np.array(Y)
feature_list = list(features.columns)
features = np.array(features)
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size =1/3, random_state = 0)

#visualistion
cab_weather.info()
sb.lmplot(data=cab_weather,x='distance',y='price',fit_reg=True,hue='location',size=10,aspect=0.5)
sb.lmplot(data=cab_weather,x='time_stamp_y',y='price',fit_reg=True,hue='location',size=10,aspect=0.5)
