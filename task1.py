import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set visual style
sns.set_theme(style='whitegrid')

# 1. Import the dataset and explore basic info
print('--- 1. LOADING DATASET ---')
df = pd.read_csv('Titanic-Dataset.csv')
print(df.info())
print('\nMissing values per column:\n', df.isnull().sum())

# 2. Handle missing values
print('\n--- 2. HANDLING MISSING VALUES ---')
# Fill missing 'Age' with the median age
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill missing 'Embarked' with the most frequent value (mode)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop 'Cabin' column due to high missing value percentage (~687 missing out of 891)
df = df.drop(columns=['Cabin'])
print("Missing values after handling:\n", df.isnull().sum())

# 3. Convert categorical features into numerical using encoding
print('\n--- 3. CATEGORICAL ENCODING ---')
# Map 'Sex' to binary values (0 and 1)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Use One-Hot Encoding for 'Embarked'
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
print(df[['Sex', 'Embarked_Q', 'Embarked_S']].head(3))

# 4. Normalize/Standardize numerical features
print('\n--- 4. FEATURE SCALING ---')
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# Standardizing 'Age' and 'Fare'
df[['Age_scaled', 'Fare_scaled']] = scaler.fit_transform(
    df[['Age', 'Fare']]
)
print(df[['Age', 'Age_scaled', 'Fare', 'Fare_scaled']].head(3))

# 5. Visualize outliers using boxplots and remove them (using IQR method on 'Fare')
print('\n--- 5. OUTLIER DETECTION & REMOVAL ---')

# Plotting boxplot before removal
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['Fare'])
plt.title('Fare Boxplot Before Outlier Removal')
plt.savefig('fare_boxplot_before.png')
plt.close()

# Calculating IQR for 'Fare'
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out outliers
df_filtered = df[(df['Fare'] >= lower_bound) & (df['Fare'] <= upper_bound)]
print(f'Rows before outlier removal: {len(df)}')
print(f'Rows after outlier removal: {len(df_filtered)}')

# Plotting boxplot after removal
plt.figure(figsize=(8, 4))
sns.boxplot(x=df_filtered['Fare'])
plt.title('Fare Boxplot After Outlier Removal')
plt.savefig('fare_boxplot_after.png')
plt.close()

# Save cleaned dataset
df_filtered.to_csv('Titanic_Cleaned.csv', index=False)
print('\nData preprocessing complete! Cleaned dataset saved as Titanic_Cleaned.csv')