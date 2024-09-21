import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the data
insurance_data = pd.read_csv("insurance_claims.csv")

# Displaying the first few rows of the dataset
print(insurance_data.head())

# Summary statistics of the dataset
summary_stats = insurance_data.describe(include='all')

# Checking for missing values
missing_values = insurance_data.isnull().sum()

print(summary_stats)

print(missing_values)

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Plotting distributions of numerical variables
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(insurance_data['age_of_driver'], kde=True, bins=15, ax=ax[0])
ax[0].set_title('Age of Driver Distribution')

sns.histplot(insurance_data['car_age'], kde=True, bins=15, ax=ax[1])
ax[1].set_title('Car Age Distribution')

sns.histplot(insurance_data['number_of_claims'], kde=False, bins=range(6), ax=ax[2])
ax[2].set_title('Number of Claims Distribution')

plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split

# One-Hot Encoding for 'region' variable
insurance_data_encoded = pd.get_dummies(insurance_data, columns=['region'], drop_first=True)

# Splitting the data into training and testing sets
train, test = train_test_split(insurance_data_encoded, test_size=0.2, random_state=42)

print(train.head())

# Define features and target
X_train = train.drop('number_of_claims', axis=1)
y_train = train['number_of_claims']

X_train_transformed = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age_of_driver', 'car_age']),
        ('cat', OneHotEncoder(drop='first'), X_train.select_dtypes(include=['object']).columns)
    ]).fit_transform(X_train)

poisson_glm = sm.GLM(y_train, sm.add_constant(X_train_transformed), family=sm.families.Poisson()).fit()

# Display the model summary
print(poisson_glm.summary())
