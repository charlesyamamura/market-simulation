import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_excel('data1319.xlsx', sheet_name='Sheet1')

# Basic Statistics
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Frequency of each model
print(data['model'].value_counts())

# Yearly Trends
yearly_share = data.groupby('year')['share'].mean()
yearly_share.plot(kind='line', title='Average Share Over Years')
plt.show()

# Monthly Trends
monthly_share = data.groupby('mo')['share'].mean()
monthly_share.plot(kind='line', title='Average Share Over Months')
plt.show()

# Correlation Matrix (only numeric columns)
numeric_data = data.select_dtypes(include=['float64', 'int64'])
corr_matrix = numeric_data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Top Models by Average Share
top_models = data.groupby('model')['share'].mean().sort_values(ascending=False)
top_models.plot(kind='bar', title='Top Models by Average Share')
plt.show()

# Attribute Comparison for Top Models
top_models_list = top_models.index[:5]
top_models_data = data[data['model'].isin(top_models_list)]
sns.boxplot(x='model', y='share', data=top_models_data)
plt.show()

# Outlier Detection
sns.boxplot(x=data['share'])
plt.show()