import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load the dataset
df = pd.read_csv('DA_MajorPROJ/bollywood_movie_analytics.csv')

# Data Preprocessing
# Display basic info
print("Dataset Shape:", df.shape)
print("\nColumn Names:\n", df.columns)
print("\nSample Data:\n", df.head())

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

#check for data types
print("\nData Types:\n", df.dtypes)



#Exploratory Data Analysis (EDA)
# Visualize the distribution of box office collection
plt.figure(figsize=(10, 6))
sns.histplot(df['BoxOffice_INR'], bins=30, kde=True, color=sns.color_palette("viridis")[2])
plt.title('Distribution of Box Office Collection')
plt.xlabel('Box Office Collection')
plt.ylabel('Frequency')
plt.show()

# Visualize the relationship between budget and box office collection
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Budget_INR', y='BoxOffice_INR', data=df, hue='Genre', palette='Set2')
plt.title('Budget vs Box Office Collection')
plt.xlabel('Budget')
plt.ylabel('Box Office Collection')
plt.show()

# Visualize the relationship between rating and box office collection
plt.figure(figsize=(10, 6))
sns.scatterplot(x='IMDb_Rating', y='BoxOffice_INR', data=df, hue='Genre', palette='coolwarm')
plt.title('Rating vs Box Office Collection')
plt.xlabel('Rating')
plt.ylabel('Box Office Collection')
plt.show()

# Visualize the distribution of genres
plt.figure(figsize=(12, 6))
df['Genre'].value_counts().plot.pie(autopct='%1.1f%%', startangle=140, cmap='tab20', figsize=(8, 8))
plt.title('Distribution of Genres')
plt.xlabel('Count')
plt.show()

# Visualize the relationship  of imdb  ratings with box office collection
plt.figure(figsize=(10, 6))
sns.lineplot(
    x='IMDb_Rating',
    y='BoxOffice_INR',
    data=df,
    estimator='mean',
    ci=None,
    color='purple',
    marker='x',
    markerfacecolor='orange',  # set marker color
    markeredgecolor='orange',  # set marker edge color
    markersize=10
)
plt.title('Average Box Office Collection by IMDb Rating')
plt.xlabel('Rating')
plt.ylabel('Box Office Collection')
plt.show()

# Visualize the distribution of  tmdb ratings
plt.figure(figsize=(10, 6))
sns.lineplot(x='TMDB_Rating', y='BoxOffice_INR', data=df, estimator='mean', ci=None, color='teal', marker='x',
    markerfacecolor='black',  # set marker color
    markeredgecolor='black',  # set marker edge color
    markersize=10)
plt.title('Average Box Office Collection by TMDB Rating')
plt.xlabel('Rating')
plt.ylabel('Box Office Collection')
plt.show()

# Visualize the distribution of languages
plt.figure(figsize=(8, 8))
df['Language'].value_counts().plot.pie(autopct='%1.1f%%', startangle=140, cmap='Pastel1')
plt.title('Distribution of Languages')
plt.show()

# Visualize the relationship between runtime and box office collection
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Duration_Minutes', y='BoxOffice_INR', data=df, hue='Genre', palette='husl')
plt.title('Runtime vs Box Office Collection')
plt.xlabel('Runtime (minutes)')
plt.ylabel('Box Office Collection')
plt.show()

#visualize the relationship between release month and box office collection
plt.figure(figsize=(10, 6))
sns.boxplot(x='Release_Month', y='BoxOffice_INR', data=df, palette='Set1')
plt.title('Release Month vs Box Office Collection')
plt.xlabel('Release Month')
plt.ylabel('Box Office Collection')
plt.xticks(rotation=45)
plt.show()

#correlation Matrix of Numerical Features
numeric_df = df.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numeric_df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Select features and target
features = ['Genre', 'Budget_INR', 'IMDb_Rating', 'TMDB_Rating', 'Duration_Minutes']
X = df[features]
y = df['BoxOffice_INR']

# One-hot encode categorical columns (like Genre)
X_encoded = pd.get_dummies(X, columns=['Genre'], drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=0)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

#visualize the predictions with actual values using a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predictions')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Actual')
plt.title('Actual vs Predicted Box Office Collection')
plt.xlabel('Actual Box Office Collection')
plt.ylabel('Predicted Box Office Collection')
plt.xlim([0, y.max()])
plt.ylim([0, y.max()])
plt.grid()
plt.legend()
plt.show()





















