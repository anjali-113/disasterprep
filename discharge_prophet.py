
# Let's start off by importing the relevant libraries
import pandas as pd
import numpy as np
import math
import itertools
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import time
import prophet
import matplotlib.pyplot as plt
import joblib

# # Loading data
def discharge_forecast(filename,wtd):
    def import_data():
        raw_data_df = pd.read_excel('data/'+filename+'.xlsx', header=0) # creates a Pandas data frame for input value
        return raw_data_df

    raw_data_df = import_data()
    raw_data_df.head()

    raw_data_df['Date']=pd.to_datetime(raw_data_df['Date'])

    for i in range(1,len(raw_data_df.columns)):
        raw_data_df[raw_data_df.columns[i]] = raw_data_df[raw_data_df.columns[i]].fillna(raw_data_df[raw_data_df.columns[i]].mean())
    data=pd.DataFrame()
    data['Date']=raw_data_df["Date"]
    data['Discharge']=raw_data_df["Discharge"]
    data=data.set_index(['Date'])
    data.head()
    data.isnull().sum()

    data.dropna().describe()

    #---------------Resampling-------------------------------
    monthly = data.resample('M').sum()
    monthly.plot(style=[':', '--', '-'],title='Monthly Trends')
    weekly = data.resample('W').sum()
    daily = data.resample('D').sum()
	#----------------------Scaling-----------------------------

	#Use MinMaxScaler to normalize  to range from 0 to 1
    values = daily['Discharge'].values.reshape(-1,1)
    values = values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    scale=daily
    scale["Discharge"]=scaled
 

	#----------Making data set for Testing or Training-------

    def making_dataset(i):
        #Testing the future prediction
        if i==0 :
            #Taking data of last two years as testing data
            df1=scale.iloc[6940:,:]
            #Training Data
            df2=scale.iloc[:6940,:]
            df2.reset_index(inplace=True)
            # Prophet requires columns ds (Date) and y (value)
            df2 = df2.rename(columns={'Date': 'ds', 'Discharge': 'y'})
            return df1,df2
        else:
            #Predicting the future values after 2018
            df2=scale.iloc[:,:]
            df2.reset_index(inplace=True)
            # Prophet requires columns ds (Date) and y (value)
            df2 = df2.rename(columns={'Date': 'ds', 'Discharge': 'y'})
            return df2,df2
    

    df1,df2=making_dataset(wtd)
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)


    #----------------------Model(FbProphet)---------------------------------


    # Make the prophet model and fit on the data
    df2_prophet = prophet.Prophet(changepoint_prior_scale=0.05)
    df2_prophet.fit(df2)
    path='trained/'+filename+'_discharge_prophet'
    joblib.dump(df2_prophet, path+'.pkl')
    df2_prophet= joblib.load(path+'.pkl')
    warnings.resetwarnings()


    #                    Making future DataFrame


    def predicting_data(i=1):
        if i==0:
            #For testing 
            # Make a future dataframe for (2 Years)
            df2_forecast = df2_prophet.make_future_dataframe(periods=30*25 , freq='D')
            # Make predictions
            df2_forecast = df2_prophet.predict(df2_forecast)
            df3=df2_forecast[['ds','yhat']]
            df3.shape,df1.shape,df2.shape
            df4=df3.iloc[6940:-20,:]

        else:
            #For Future prediction of 2019
            # Make a future dataframe for 12 months
            df2_forecast = df2_prophet.make_future_dataframe(periods=30*12 , freq='D',include_history=False)
            # Make predictions
            df2_forecast = df2_prophet.predict(df2_forecast)
            df3=df2_forecast[['ds','yhat']]
            #df3.shape,df1.shape,df2.shape
            df4=df3.iloc[:,:]
        return df4,df2_forecast


    df4,df2_forecast=predicting_data(wtd)
    ypred=df4.iloc[:,1:]
    ytest=df1.iloc[:,:]
    ypred.shape,ytest.shape

    df4.tail()
    ypred=df4.iloc[:,1:]
    ytest=df1.iloc[:,:]
    ypred.shape,ytest.shape

    from sklearn.metrics import mean_absolute_error
    if wtd==0:
        print("mean_absolute_error=",mean_absolute_error(ytest,ypred))


    #Future data plot		
    df2_prophet.plot(df2_forecast, xlabel = 'Date', ylabel = 'Discharge')
    plt.title('simple test');

    # Plot the trends and patterns
    df2_prophet.plot_components(df2_forecast)
    df4.columns = ['Date', 'Discharge']

    #Getting the vaues in original range
    values = df4['Discharge'].values.reshape(-1,1)
    values = values.astype('float32')
    valu = scaler.inverse_transform(values)
    df4['Discharge']=valu
    df4['Discharge']= abs(df4['Discharge'])
    df4.to_csv('data/forecast/'+filename+'_discharge_forecast.csv',index=False)

    return df4 






