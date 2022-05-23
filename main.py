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
df.neighborhood.unique()
df.rent.describe()
df.hist(grid=False, figsize=(15,15), layout=(5,3), color='#6FD1DF');
keys = list(df.city.value_counts().keys())
vals = list(df.city.value_counts())
sns.set_theme(style="darkgrid")

f, ax = plt.subplots(figsize=(12, 8))

ax.set_title('Total Rental Units by Neighborhood',
             fontname='silom', fontsize=15)

ax.set_xlabel('Count',
             fontname='silom', fontsize=12)

ax.set_ylabel('Neighborhood',
             fontname='silom', fontsize=12)

sns.barplot(x=vals, y=keys,
            color="rebeccapurple");

avg_rents = df.groupby('neighborhood').mean().reset_index()[['neighborhood', 'rent']]
avg_rents['rent'] = [round(r) for r in avg_rents.rent]
avg_rents.sort_values(by=['rent'], ascending=False, inplace=True)
order = list(avg_rents.neighborhood)
plt.figure(figsize=((17, 10)))
ax = sns.boxplot(x="neighborhood", y="rent",
                 data=df, linewidth=2.5,
                 order=order)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.xlabel("Neigborhood", size=15, fontname='silom')
plt.ylabel("Rent", size=15, fontname='silom')
plt.title("Manhattan Rental Costs by Neighborhood", size=18, fontname='silom');

neighborhood_dummies = pd.get_dummies(df.neighborhood).drop('Manhattanville', axis=1)
df = pd.concat([df, neighborhood_dummies], axis=1).drop('neighborhood', axis=1)
scatter_matrix (df, figsize = (40,40), alpha = 0.9, diagonal = "kde", marker = "o");

df_correlated = df[['rent', 'bedrooms', 'bathrooms', 'size_sqft']]
scatter_matrix(df_correlated, figsize = (16,10),
               alpha = 0.9, diagonal = "kde", marker = "o")

plt.suptitle('Scatter Matrix of Features Correlated with Rent',
          fontsize=16,
          fontname='silom')
y = df.rent
x = np.array(df['size_sqft']).reshape(-1, 1)
model = LinearRegression()
model.fit(x, y)

residuals = y - model.predict(x)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,7))
residuals = y - model.predict(x)
sns.histplot(residuals, ax=ax1, kde=True)
ax1.set(xlabel="Residuals")
sm.qqplot(residuals, stats.t, distargs=(4,), fit=True, line="45", ax=ax2);
plt.suptitle('Linear Model Residuals', fontname='silom', fontsize=15);

fig = plt.figure(figsize=(15,7))
plt.scatter(model.predict(x), residuals, color='rebeccapurple')
plt.axhline(y=0, color='lightgreen', linestyle='-', linewidth=4)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Fitted Vs. Residuals', fontname='silom', fontsize=15)
ax = plt.gca()
ax.set_facecolor('#f9f6fc')
plt.show()
plt.show()

def predict_rent(sqft):
    return model.predict(np.array([sqft]).reshape(1, -1))
predict_rent(250)

predict_rent(500)

1223*2

print('R-Squared:', round(model.score(x, y), 2))
print('Coefficient:', round(model.coef_[0], 2))
print('Intercept:', round(model.intercept_, 2))