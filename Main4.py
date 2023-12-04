import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import scipy.stats as sc

df = pd.read_csv('US_Accidents_March23.csv')

sevdf = df["Severity"]
x = sevdf
y = df["Precipitation(in)"]
plt.plot(x, y, 'o')
plt.show()

"""import sklearn.linear_model as lm

from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()
x = x.to_numpy()
x = x.reshape(-1,1)
linear_model.fit(x,y)
y_hat = linear_model.predict(x)
plt.plot(x,y,'o')
plt.plot(x,y_hat,'r')"""

# Create replica of dataframe
cond = df

# Reset index while extrapolating Severity values into lists
cond_lists = cond.groupby("Weather_Condition")["Severity"].apply(list).reset_index()

# Compare unique weather conditions
conditions = cond_lists['Weather_Condition'].unique()

pval = []

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

        alpha = 0.05

        # Checks statistical significance (if p < 0.05)
        if p < alpha:
            #print(f"T-test between {condition1} and {condition2}:")
            #print(f"P-value: {p}")
            #print()
            # Append values to list to print later
            pval.append(p)

# Print first 5 values of sorted list
pval.sort()
print(f"{pval[:5]}")