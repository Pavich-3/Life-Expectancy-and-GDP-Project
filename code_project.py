import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import trim_mean, iqr, pearsonr
import statsmodels.api as sm

df = pd.read_csv('all_data.csv')
print(df.head())
print(df.dtypes)
print(df.describe())
print(df.Country.unique())
print(df.Year.unique())

# Has life expectancy increased over time in the six nations?

plt.figure("Life expectancy at birth (years)", figsize=(10, 10))
sns.set_theme(style="whitegrid")

plt.subplot(3, 2, 1)
sns.lineplot(x=df.Year, y='Life expectancy at birth (years)', hue=df['Country'][df.Country == 'Chile'], data=df, linewidth=2.5)
plt.ylabel('LEB (year)')

plt.subplot(3, 2, 2)
sns.lineplot(x=df.Year, y='Life expectancy at birth (years)', hue=df['Country'][df.Country == 'China'], data=df, linewidth=2.5)
plt.ylabel('LEB (year)')

plt.subplot(3, 2, 3)
sns.lineplot(x=df.Year, y='Life expectancy at birth (years)', hue=df['Country'][df.Country == 'Germany'], data=df, linewidth=2.5)
plt.ylabel('LEB (year)')

plt.subplot(3, 2, 4)
sns.lineplot(x=df.Year, y='Life expectancy at birth (years)', hue=df['Country'][df.Country == 'Mexico'], data=df, linewidth=2.5)
plt.ylabel('LEB (year)')

plt.subplot(3, 2, 5)
sns.lineplot(x=df.Year, y='Life expectancy at birth (years)', hue=df['Country'][df.Country == 'United States of America'], data=df, linewidth=2.5)
plt.ylabel('LEB (year)')

plt.subplot(3, 2, 6)
sns.lineplot(x=df.Year, y='Life expectancy at birth (years)', hue=df['Country'][df.Country == 'Zimbabwe'], data=df, linewidth=2.5)
plt.ylabel('LEB (year)')

sns.color_palette("husl", 9)
plt.subplots_adjust(wspace=0.3, hspace=0.5)
plt.show()
plt.clf()

# Has GDP increased over time in the six nations?

plt.figure("GDP", figsize=(10, 10))
sns.set_theme(style="whitegrid")

df_three = df[(df.Country=='China') | (df.Country=='United States of America') | (df.Country=='Germany')].reset_index()
df_other_three = df[(df.Country=='Chile') | (df.Country=='Zimbabwe') | (df.Country=='Mexico')].reset_index()

ax = plt.subplot(2, 1, 1)
sns.lineplot(x='Year', y='GDP', data=df_three, linewidth=2.5, hue='Country')
plt.title("GDP about country")
ax.get_xaxis().get_major_formatter()

plt.subplot(2, 1, 2)
sns.lineplot(x='Year', y='GDP', data=df_other_three, linewidth=2.5, hue='Country')
plt.title("GDP about country")

sns.color_palette("tab10")
plt.subplots_adjust(hspace=0.3)
plt.show()
plt.clf()

# Is there a correlation between GDP and life expectancy of a country?

corr, p = pearsonr(df['Life expectancy at birth (years)'], df.GDP)
print(np.round(corr, 2))

# What is the average life expectancy in these nations?

avg_chile = np.round(df[df.Country == 'Chile'].iloc[0:, 2].mean(), 2)
print(f"The average life expectancy in Chile: {avg_chile}.")

avg_china = np.round(df[df.Country == 'China'].iloc[0:, 2].mean(), 2)
print(f"The average life expectancy in China: {avg_china}.")

avg_germany = np.round(df[df.Country == 'Germany'].iloc[0:, 2].mean(), 2)
print(f"The average life expectancy in Germany: {avg_germany}.")

avg_mexico = np.round(df[df.Country == 'Mexico'].iloc[0:, 2].mean(), 2)
print(f"The average life expectancy in Mexico: {avg_mexico}.")

avg_usa = np.round(df[df.Country == 'United States of America'].iloc[0:, 2].mean(), 2)
print(f"The average life expectancy in the USA: {avg_usa}.")

avg_zimbabwe = np.round(df[df.Country == 'Zimbabwe'].iloc[0:, 2].mean(), 2)
print(f"The average life expectancy in the Zimbabwe: {avg_zimbabwe}.")

avg_all = np.round(df.iloc[0:, 2].mean(), 2)
print(f"The average life expectancy in all the countries: {avg_all}.")

list_life = [avg_chile, avg_china, avg_germany, avg_mexico, avg_usa, avg_zimbabwe]
array_list = np.array(list_life)
list_country = list(df.Country.unique())
array_country = np.array(list_country)

plt.figure("The average life expectancy", figsize=(10, 5))
sns.barplot(x=array_country, y=array_list, palette='pastel')
plt.title('The average life expectancy')
plt.ylabel('The average life expectancy')
plt.xlabel('Country')

plt.show()
plt.clf()

# Linear Regression using statsmodels

df.columns = ['Country', 'Year', 'Life', 'GDP']
model = sm.OLS.from_formula('Life ~ GDP', data=df)
results = model.fit()
print(results.summary())
print(results.params)
fitted_values = results.predict(df)
residuls = df.Life - fitted_values

plt.subplot(2, 2, 1)
sns.scatterplot(x='Life', y='GDP', data=df, hue='Country')

plt.subplot(2, 2, 2)
plt.hist(residuls)

plt.subplot(2, 2, 3)
sns.scatterplot(x=fitted_values, y=residuls, hue='Country')

plt.show()
plt.clf()

# GDP for each country

df_chile = df[df.Country == 'Chile']
df_china = df[df.Country == 'China']
df_germany = df[df.Country == 'Germany']
df_mexico = df[df.Country == 'Mexico']
df_usa = df[df.Country == 'United States of America']
df_zimbabwe = df[df.Country == 'Zimbabwe']

plt.figure("GDP Country", figsize=(10, 6))

plt.subplot(3, 3, 1)
sns.lineplot(x='Year', y='GDP', data=df_chile, color='black')
plt.title("GDP Chile")

plt.subplot(3, 3, 2)
sns.lineplot(x='Year', y='GDP', data=df_china, color='orange')
plt.title("GDP China")

plt.subplot(3, 3, 3)
sns.lineplot(x='Year', y='GDP', data=df_germany, color='grey')
plt.title("GDP Germany")

plt.subplot(3, 3, 4)
sns.lineplot(x='Year', y='GDP', data=df_mexico, color='green')
plt.title("GDP Mexico")

plt.subplot(3, 3, 5)
sns.lineplot(x='Year', y='GDP', data=df_usa, color='blue')
plt.title("GDP the USA")

plt.subplot(3, 3, 6)
sns.lineplot(x='Year', y='GDP', data=df_zimbabwe, color='brown')
plt.title("GDP Zimbabwe")

plt.subplots_adjust(wspace=0.3, hspace=0.6)
plt.show()
plt.clf()
