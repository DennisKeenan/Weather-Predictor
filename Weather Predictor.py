import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plot

# Read and Edit Data
    # Read Data
data=pd.read_csv("weather.csv")
# print(data.shape)
# print(data.columns)
# print(data.info())
# print(data["Location"].unique())

    # Read Data (Specified)
# text=[i for i in data.columns if data[i].dtype=='O']
# print(data[text].head())
# print(data[text].isnull().sum())
# for i in text:
    # print(data[i].value_counts())
    # print(i,": ",len(data[i].unique()))

    # Edit Data
data["Date"]=pd.to_datetime(data["Date"])
data["Year"]=data["Date"].dt.year
data["Month"]=data["Date"].dt.month
data["Day"]=data["Date"].dt.day
data.drop("Date",axis=1,inplace=True)