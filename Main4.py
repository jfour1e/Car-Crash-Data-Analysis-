# Libraries to import
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import scipy.stats as sc
import random

# Read CSV file
df = pd.read_csv('US_Accidents_March23.csv')

# Create replica of dataframe
cond = df

# Reset index while extrapolating Severity values into lists
cond_lists = cond.groupby("Weather_Condition")["Severity"].agg(list).reset_index()

# Compare unique weather conditions
conditions = cond_lists['Weather_Condition'].unique()

# Initiate dictionary
pval = {}

for i in range(len(conditions)):
    condition1 = conditions[i]

    # Extract severity values list for condition1
    severity1 = cond_lists.loc[cond_lists['Weather_Condition'] == condition1, 'Severity'].values[0]

    for j in range(i + 1, len(conditions)):
        condition2 = conditions[j]

        # Extract severity values list for condition2
        severity2 = cond_lists.loc[cond_lists['Weather_Condition'] == condition2, 'Severity'].values[0]

        # Perform t-test
        _, p = sc.ttest_ind(severity1, severity2, nan_policy='omit')

        # Add results of t-test to dictionary
        mystr = "Mean severity of car crashes in " + condition1 + " weather as compared to mean severity of car crashes in " + condition2 + "weather"
        pval[mystr] = p

# Randomly generate 5 different results from the dictionary and show conclusion of presence of statistical signiicance
for _ in range(5):
    ran = random.choice(list(pval.keys()))
    print(f'Random Comparison: {ran}, P - Value: {pval[ran]}')
    if pval[ran] < 0.05:
        print(f"There is statistical significance in the mean severity of car crashes based on the weather condition change from {condition1} to {condition2}")
    else:
        print(f"There is no statistical significance in the mean severity of car crashes based on the weather condition change from {condition1} to {condition2}")

# Create replica of dataframe
traffic = df

# Initiate variables needed for chi-squared test
fourTrue = 0
fourFalse = 0
oneTrue = 0
oneFalse = 0

# Filter/Make new dataframe including values of severity that only equal 1 or 4, as well as taking the whole "Traffic_Signal" column
tradf = traffic[(traffic["Severity"] == 1) | (traffic["Severity"] == 4)][["Severity", "Traffic_Signal"]].copy()

# Iterate through index/row of new dataframe, "tradf"
for i, r in tradf.iterrows():
    sev = r["Severity"]
    trafficSignal = r["Traffic_Signal"]
    # Add to count for variables if conditions hold true/false
    if sev == 4 and trafficSignal == True:
        fourTrue += 1
    elif sev == 4 and trafficSignal == False:
        fourFalse += 1
    elif sev == 1 and trafficSignal == True:
        oneTrue += 1
    else:
        oneFalse += 1

# Put data together into a 2D array
myData = [[fourTrue, fourFalse], [oneTrue, oneFalse]]
# Perform chi-squared test, analyze p value, and make conclusion based off of value
_, p, _, _ = sc.chi2_contingency(myData)
print(f"The P - Value of this chi-squared test comparing the presence of a traffic signal to the relationship of either a severity level of 1 or 4 occurring is {p}.")
if (p < 0.05):
  print(f"Because the P - Value of {p} is less than 0.05, this result is Statistically Significant.")
else:
  print(f"Because the P - Value of {p} is greater than or equal to 0.05, this result is not Statistically Significant.")



"""
sevdf = df["Severity"]
x = sevdf
y = df["Precipitation(in)"]
plt.plot(x, y, 'o')
plt.show()

import sklearn.linear_model as lm

from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()
x = x.to_numpy()
x = x.reshape(-1,1)
linear_model.fit(x,y)
y_hat = linear_model.predict(x)
plt.plot(x,y,'o')
plt.plot(x,y_hat,'r')
"""