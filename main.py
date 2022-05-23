import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
import statsmodels.api as sm
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_cs('//Users/joy gichure/downloads/nairobioffice.csv')
df.head()

df.info()
df.describe()
df.borough.unique()

df.drop('borough', axis=1, inplace=True)
df.drop('office_id', axis=1, inplace=True)
df.isna().sum()
df.size.unique()
df.price.describe()
df.hist(grid=False, figsize=(15, 15), layout=(5, 3), color='#6FD1DF');
keys = list(df.city.value_counts().keys())
vals = list(df.city.value_counts())
sns.set_theme(style="darkgrid")

f, ax = plt.subplots(figsize=(12, 8))

ax.set_title('Total offices',
             fontname='silom', fontsize=15)

ax.set_xlabel('size',
              fontname='silom', fontsize=12)

ax.set_ylabel('price',
              fontname='silom', fontsize=12)

sns.barplot(x=vals, y=keys,
            color="rebeccapurple");

avg_rents = df.groupby('size').mean().reset_index()[['size', 'price']]
avg_rents['price'] = [round(r) for r in avg_rents.rent]
avg_rents.sort_values(by=['price'], ascending=False, inplace=True)
order = list(avg_rents.neighborhood)
plt.figure(figsize=((17, 10)))
ax = sns.boxplot(x="size", y="price",
                 data=df, linewidth=2.5,
                 order=order)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.xlabel("Size", size=15, fontname='silom')
plt.ylabel("Price", size=15, fontname='silom')
plt.title("office price by square area", size=18, fontname='silom');

plt.suptitle('Scatter Matrix based on prices',
             fontsize=16,
             fontname='silom')
y = df.price
x = np.array(df['size_sqft']).reshape(-1, 1)
model = LinearRegression()
model.fit(x, y)

residuals = y - model.predict(x)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
residuals = y - model.predict(x)
sns.histplot(residuals, ax=ax1, kde=True)
ax1.set(xlabel="Residuals")
sm.qqplot(residuals, stats.t, distargs=(4,), fit=True, line="45", ax=ax2);
plt.suptitle('Linear Model Residuals', fontname='silom', fontsize=15);

fig = plt.figure(figsize=(15, 7))
plt.scatter(model.predict(x), residuals, color='rebeccapurple')
plt.axhline(y=0, color='lightgreen', linestyle='-', linewidth=4)
plt.xlabel('Predictedprice')
plt.ylabel('Residuals')
plt.title('Fitted Vs. Residuals', fontname='silom', fontsize=15)
ax = plt.gca()
ax.set_facecolor('#f9f6fc')
plt.show()
plt.show()


def predict_price(area):
    return model.predict(np.array([sqft]).reshape(1, -1))


predict_price(100)
print('R-Squared:', round(model.score(x, y), 2))
print('Coefficient:', round(model.coef_[0], 2))
print('Intercept:', round(model.intercept_, 2))
