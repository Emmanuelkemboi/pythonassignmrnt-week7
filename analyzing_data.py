# analyzing_data.py this is my  response t the asssignent 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris


# Task 1: Load and explore Dataset


try:
    # Load Iris dataset from sklearnn
    iris = load_iris(as_frame=True)
    df = iris.frame
    df['species'] = df['target'].map(dict(enumerate(iris.target_names)))

    print("First five rows of the dataset:")
    print(df.head())

    print("\nDataset Info:")
    print(df.info())

    print("\nChecking for missing values:")
    print(df.isnull().sum())



except FileNotFoundError as e:
    print("Error loading dataset:", e)


# Task 2: Basic Data Analysi

print("\nBasic statistics of numerical columns:")
print(df.describe())

print("\nMean values grouped by species:")
group_means = df.groupby("species").mean()
print(group_means)

# Interesting finding
print("\nObservation: Setosa flowers tend to have smaller sepal and petal dimensions compared to Virginica.")

# Task 3: Data Visualization
# this is for te visuaisatio of te dara

# 1. Line Chart - sepal length over index
plt.figure(figsize=(8, 5))
plt.plot(df.index, df['sepal length (cm)'], label="Sepal Length", color="blue")
plt.title("Line Chart: Sepal Length Over Index")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.grid(True)
plt.show()

# 2. Bar Chart - avarage petal length per species
plt.figure(figsize=(8, 5))
group_means['petal length (cm)'].plot(kind='bar', color=["green", "orange", "red"])
plt.title("Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# 3. Histogram - distributio of sepal width
plt.figure(figsize=(8, 5))
plt.hist(df['sepal width (cm)'], bins=15, color="purple", edgecolor="black")
plt.title("Histogram: Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

#  Scattter Plot - sepal length vs petal length
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="sepal length (cm)", y="petal length (cm)", hue="species", palette="deep")
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()
