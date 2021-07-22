import io
import streamlit as st
import datetime
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import os
import matplotlib.pyplot as plt
from fbprophet import Prophet
from pandas import to_datetime
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
    
# os.chdir('C:/Users/FarzanehAkhbar/Documents/FAAS/bitcoin/New folder/bitcoin-predict-master/data')


st.header("Prophet Model for Time Series Prediction")


in_file = st.file_uploader("Choose a CSV file", accept_multiple_files=False, type='csv')
if in_file is not None:
    # data = pd.read_csv(io.StringIO(in_file.read().decode('utf-8')), sep=',', index_col=0)
    data = pd.read_csv(in_file)

    # data.index = pd.to_datetime(data.index)
    # graph = util.make_timeseries_graph(data, title="Complete Timeseries Data")
    # st.markdown(f"Name: **{in_file.name}**, Datapoints: **{len(data)}**, Date Range: **{data.index[0].date()}** - **{data.index[-1].date()}**")
    # st.pyplot(graph)

# ## Data Exploration
# data = pd.read_csv("bitcoin.csv")


def functest(data):
    data = data.sort_values('Date')
    data['Date']= pd.to_datetime(data['Date'])
    
    df = data.drop(['Symbol', 'Open', 'High', 'Low', 'Volume BTC', 'Volume USD'],1)
    
    
    
    min_max_scaler = MinMaxScaler()
    norm_data = pd.DataFrame(min_max_scaler.fit_transform(df[['Close']].values))
    norm_data.columns = ['val']
    df['x'] = norm_data['val']
    df = df.drop(['Close'],1)
    
    
    df.columns = ['ds','y']
    
    
    #asking user for test/train split threshold
    tx1 = '<p style="font-family:Courier; color:Blue; font-size: 20px;">Enter test/train split threshold:</p>'
    st.markdown(tx1, unsafe_allow_html=True)
    threshold = st.number_input("", value=0.8, step=0.1)
    # threshold = 0.97
    TRAIN_SPLIT = int(len(df) * (threshold))
    st.markdown(f"threshold: **{threshold}**")
    
    # tx3 = '<p style="font-family:Courier; color:Blue; font-size: 20px;">Enter Future Prediction Horizon: </p>'
    # st.markdown(tx3, unsafe_allow_html=True)
    # num_prediction = st.number_input("", value=15)
    
    
    
    tx = '<p style="font-family:Courier; color:Blue; font-size: 20px;">Enter Required Confidence Interval:  </p>'
    st.markdown(tx, unsafe_allow_html=True)
    confidencee = st.number_input("", value= 0.85, step=0.1)
    
    
    train = df.iloc[:TRAIN_SPLIT,:]
    test = df.iloc[TRAIN_SPLIT:,:]
    m = Prophet(interval_width= confidencee)
    m.fit(train)
    
    
    y = pd.DataFrame(data['Close'].iloc[TRAIN_SPLIT:])
    testy = test.drop(['y'],1)
    
    # future = m.make_future_dataframe(periods=100)
    forecast = m.predict(testy)
    res = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    # forecast['yhat'] = abs(forecast['yhat'])
    
    # yhat = pd.DataFrame(min_max_scaler.inverse_transform(forecast[['yhat']]))
    
    rmse = (mean_squared_error(y, forecast.yhat))**(1/2)
    st.markdown("RMSE: ") 
    rmse
    
    # m.plot(forecast)
    
    
    fig, ax = plt.subplots()
    ax.plot(test.ds, forecast.yhat, color='tab:blue', label='Actual')
    ax.plot(test.ds, y,  color='tomato', label='Prediction')
    ax.set_xlabel('Date', )
    ax.set_ylabel('Price', )
    ax.set_title('Bitcoin price',  fontweight='bold')
    ax.grid(True)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    st.pyplot(fig)
    
    # plot expected vs actual
    # pyplot.plot(y, label='Actual')
    # pyplot.plot(forecast.yhat, label='Predicted')
    # pyplot.legend()
    # pyplot.show()
    
    
    # fig, ax = plt.subplots(figsize=(12, 9))
    # ax.plot( y, label='Actual')
    # ax.plot( forecast.yhat, label='Predicted')
    # ax.set_xlabel('Date', )
    # ax.set_ylabel('Price', )
    # ax.set_title('Bitcoin price',  fontweight='bold')
    # ax.grid(True)
    # st.pyplot(fig)

    
if st.button("Press me after loading the data ^_^"):
    functest(data)
    
    