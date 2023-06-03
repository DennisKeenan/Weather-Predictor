import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plot

# Read Data
data=pd.read_csv("weather.csv")
print(data.shape)
print(data.head(10))
print(data["Location"].unique())