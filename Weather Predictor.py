import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plot

# Read Data
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


# Data Observation
    # Date (Edit)
data["Date"]=pd.to_datetime(data["Date"])
data["Year"]=data["Date"].dt.year
data["Month"]=data["Date"].dt.month
data["Day"]=data["Date"].dt.day
data.drop("Date",axis=1,inplace=True)
#     # Location
# print(data["Location"].value_counts())
# print(pd.get_dummies(data.Location,drop_first=True).head())
# print("-"*80)
#     # Wind Direction
# print(data["WindGustDir"].value_counts())
# print(pd.get_dummies(data.WindGustDir,drop_first=True,dummy_na=True).sum(axis=0))
# print(pd.get_dummies(data.WindGustDir,drop_first=True,dummy_na=True).head())
# print("-"*80)
#     Wind Direction (9am)
# print(data["WindDir9am"].value_counts()) 
# print(pd.get_dummies(data.WindDir9am,drop_first=True,dummy_na=True).sum(axis=0))
# print(pd.get_dummies(data.WindDir9am,drop_first=True,dummy_na=True).head())
# print("-"*80)
#     Wind Direction (3pm)
# print(data["WindDir3pm"].value_counts())
# print(pd.get_dummies(data.WindDir3pm,drop_first=True,dummy_na=True).sum(axis=0))
# print(pd.get_dummies(data.WindDir3pm,drop_first=True,dummy_na=True).head())
# print("-"*80)



# Data Plotting
numerical=[i for i in data.columns if data[i].dtype!='O']
print(data[numerical].isnull().sum())
plot.figure(figsize=(15,10))

    # Boxplot
        # Plot 1
plot.subplot(2,2,1)
fig=data.boxplot(column="Rainfall")
fig.set_title("Rainfall")
        # Plot 2
plot.subplot(2,2,2)
fig=data.boxplot(column="Evaporation")
fig.set_title("Evaporation")
        # Plot 3
plot.subplot(2,2,3)
fig=data.boxplot(column="WindSpeed9am")
fig.set_title("Wind Speed 9am")
        # Plot 4
plot.subplot(2,2,4)
fig=data.boxplot(column="WindSpeed3pm")
fig.set_title("Wind Speed 3pm")

    # Histogram
        # Plot 1
plot.subplot(2,2,1)
fig=data.Rainfall.hist(bins=10)
fig.set_xlabel("Rainfall")
fig.set_ylabel("Rain Tommorow")
        # Plot 2
plot.subplot(2,2,2)
fig=data.Evaporation.hist(bins=10)
fig.set_xlabel("Evaporation")
fig.set_ylabel("Rain Tommorow")
        # Plot 3
plot.subplot(2,2,3)
fig=data.WindSpeed9am.hist(bins=10)
fig.set_xlabel("Wind Speed 9am")
fig.set_ylabel("Rain Tommorow")
        # Plot 4
plot.subplot(2,2,4)
fig=data.WindSpeed3pm.hist(bins=10)
fig.set_xlabel("Wind Speed 3pm")
fig.set_ylabel("Rain Tommorow")

plot.show()

