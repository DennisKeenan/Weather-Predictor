import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split

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
#     print(data[i].value_counts())
#     print(i,": ",len(data[i].unique()))


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


# Find outliers for Rainfall variable
# print(round(data[numerical].describe()),2)

        # Rainfall
IQR = data.Rainfall.quantile(0.75) - data.Rainfall.quantile(0.25)
Lower_fence = data.Rainfall.quantile(0.25) - (IQR * 3)
Upper_fence = data.Rainfall.quantile(0.75) + (IQR * 3)
print('Rainfall outliers are values < {lowerboundary} or > {upperboundary}'
      .format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

        # Evaporation
IQR = data.Evaporation.quantile(0.75) - data.Evaporation.quantile(0.25)
Lower_fence = data.Evaporation.quantile(0.25) - (IQR * 3)
Upper_fence = data.Evaporation.quantile(0.75) + (IQR * 3)
print('Evaporation outliers are values < {lowerboundary} or > {upperboundary}'
      .format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

        # Wind Speed 9am
IQR = data.WindSpeed9am.quantile(0.75) - data.WindSpeed9am.quantile(0.25)
Lower_fence = data.WindSpeed9am.quantile(0.25) - (IQR * 3)
Upper_fence = data.WindSpeed9am.quantile(0.75) + (IQR * 3)
print('Wind Speed (9am) outliers are values < {lowerboundary} or > {upperboundary}'
      .format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

        # Wind Speed 3pm
IQR = data.WindSpeed3pm.quantile(0.75) - data.WindSpeed3pm.quantile(0.25)
Lower_fence = data.WindSpeed3pm.quantile(0.25) - (IQR * 3)
Upper_fence = data.WindSpeed3pm.quantile(0.75) + (IQR * 3)
print('Wind Speed (3pm) outliers are values < {lowerboundary} or > {upperboundary}'
      .format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

# plot.show()

# Training
x=data.drop(["RainTomorrow"],axis=1)
y=data["RainTomorrow"]  
categorical=[i for i in x.columns if x[i].dtype=='O']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
for data1 in [X_train,X_test]:
    for col in numerical:
        col_median=X_train[col].median()
        data1[col].fillna(col_median,inplace=True)
for data2 in [X_train,X_test]:
    data2["WindGustDir"].fillna(X_train["WindGustDir"].mode()[0],inplace=True)
    data2["WindDir9am"].fillna(X_train["WindDir9am"].mode()[0],inplace=True)
    data2["WindDir3pm"].fillna(X_train["WindDir3pm"].mode()[0],inplace=True)
    data2["RainToday"].fillna(X_train["RainToday"].mode()[0],inplace=True)
def max_value(data3,variable,top):
    return np.where(data3[variable]>top,top,data3[variable])
for data3 in [X_train,X_test]:
    data3["Rainfall"]=max_value(data3,"Rainfall",3.2)
    data3["Evaporation"]=max_value(data3,"Evaporation",21.8)
    data3["WindSpeed9am"]=max_value(data3,"WindSpeed9am",55.0)
    data3["WindSpeed3pm"]=max_value(data3,"WindSpeed3pm",57.0)
print(X_train[numerical].isnull().sum())
print(X_train[categorical].isnull().sum())
print(X_train.describe())
