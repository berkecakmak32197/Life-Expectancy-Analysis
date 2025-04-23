# Life Expectancy and Air Quality Analysis
# DSA 210 Introduction to Data Science Project

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Create directory for images if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

# Load the datasets
life_exp_data = pd.read_csv('Life Expectancy Data.csv')
air_quality_data = pd.read_csv('data_date.csv')

print(f"Loaded data: {life_exp_data.shape[0]} life expectancy records and {air_quality_data.shape[0]} air quality records")

# Display sample data from each dataset
print("\nSample of life expectancy data:")
print(life_exp_data.head(3))

print("\nSample of air quality data:")
print(air_quality_data.head(3))

# Determine the time range of each dataset
print(f"\nLife expectancy data covers: {min(life_exp_data['Year'])} to {max(life_exp_data['Year'])}")

# Convert dates to extract year information
air_quality_data['Date'] = pd.to_datetime(air_quality_data['Date'])
air_quality_data['Year'] = air_quality_data['Date'].dt.year
print(f"Air quality data covers: {min(air_quality_data['Year'])} to {max(air_quality_data['Year'])}")

# Data cleaning and preparation
# Remove extra spaces from column names
life_exp_data.columns = [col.strip() for col in life_exp_data.columns]

# Select only the most recent year (2015) for analysis
life_exp_2015 = life_exp_data[life_exp_data['Year'] == 2015].copy()
print(f"\nSelected data from {len(life_exp_2015)} countries in 2015 for analysis")

# Calculate average AQI for each country
country_avg_aqi = air_quality_data.groupby('Country')['AQI Value'].mean().reset_index()
country_avg_aqi.rename(columns={'AQI Value': 'Average AQI'}, inplace=True)

# Determine the most common air quality status for each country
country_status = air_quality_data.groupby('Country')['Status'].agg(
    lambda x: x.value_counts().index[0]
).reset_index()
country_status.rename(columns={'Status': 'Predominant Air Quality'}, inplace=True)

# Combine air quality information
air_quality_by_country = pd.merge(country_avg_aqi, country_status, on='Country')

# Create categorization for AQI values
def categorize_aqi(aqi):
    if aqi <= 50:
        return 'Low Risk'
    elif aqi <= 100:
        return 'Moderate Risk'
    else:
        return 'High Risk'

air_quality_by_country['AQI Risk Category'] = air_quality_by_country['Average AQI'].apply(categorize_aqi)

# Merge life expectancy and air quality datasets
combined_data = pd.merge(life_exp_2015, air_quality_by_country, on='Country', how='inner')
print(f"After merging, the dataset contains {len(combined_data)} countries")

# Display sample of merged data
print("\nSample of merged data:")
print(combined_data[['Country', 'Life expectancy', 'Average AQI', 'AQI Risk Category']].head())

# Exploratory Data Analysis
# Calculate correlation between AQI and life expectancy
correlation = combined_data['Life expectancy'].corr(combined_data['Average AQI'])
print(f"\nCorrelation between Life Expectancy and Average AQI: {correlation:.4f}")
print("(A negative correlation indicates countries with worse air quality tend to have lower life expectancy)")

# Calculate average life expectancy by AQI risk category
print("\nAverage life expectancy by AQI risk category:")
for category in ['Low Risk', 'Moderate Risk', 'High Risk']:
    avg = combined_data[combined_data['AQI Risk Category'] == category]['Life expectancy'].mean()
    print(f"- {category}: {avg:.2f} years")

# Calculate average life expectancy by development status
print("\nAverage life expectancy by development status:")
for status in ['Developed', 'Developing']:
    avg = combined_data[combined_data['Status'] == status]['Life expectancy'].mean()
    print(f"- {status}: {avg:.2f} years")

# Identify countries with highest and lowest life expectancy
top_countries = combined_data.nlargest(5, 'Life expectancy')
bottom_countries = combined_data.nsmallest(5, 'Life expectancy')

print("\nTop 5 countries with highest life expectancy:")
for i, row in top_countries.iterrows():
    print(f"- {row['Country']}: {row['Life expectancy']:.1f} years (AQI: {row['Average AQI']:.1f})")

print("\nBottom 5 countries with lowest life expectancy:")
for i, row in bottom_countries.iterrows():
    print(f"- {row['Country']}: {row['Life expectancy']:.1f} years (AQI: {row['Average AQI']:.1f})")

# Data Visualization
# 1. Scatter plot: Life Expectancy vs AQI
plt.figure(figsize=(10, 6))
plt.scatter(combined_data['Average AQI'], combined_data['Life expectancy'], color='blue', alpha=0.7)

# Add trend line
z = np.polyfit(combined_data['Average AQI'], combined_data['Life expectancy'], 1)
p = np.poly1d(z)
plt.plot(combined_data['Average AQI'], p(combined_data['Average AQI']), "r--", alpha=0.7)

plt.title('Life Expectancy vs. Air Quality Index')
plt.xlabel('Average Air Quality Index (Higher = More Pollution)')
plt.ylabel('Life Expectancy (Years)')
plt.grid(True, alpha=0.3)
plt.savefig('images/life_expectancy_vs_aqi.png')
plt.close()

# 2. Bar chart: Life expectancy by AQI risk category
plt.figure(figsize=(10, 6))
aqi_categories = ['Low Risk', 'Moderate Risk', 'High Risk']
life_exp_by_risk = []

for cat in aqi_categories:
    avg = combined_data[combined_data['AQI Risk Category'] == cat]['Life expectancy'].mean()
    life_exp_by_risk.append(avg)
    # Add value labels on bars
    plt.text(aqi_categories.index(cat), avg + 0.5, f"{avg:.1f}", ha='center')

plt.bar(aqi_categories, life_exp_by_risk, color='lightblue')
plt.title('Average Life Expectancy by Air Quality Risk Category')
plt.xlabel('Air Quality Risk Category')
plt.ylabel('Average Life Expectancy (Years)')
plt.grid(axis='y', alpha=0.3)
plt.savefig('images/life_expectancy_by_risk.png')
plt.close()

# 3. Box plot: Life expectancy by development status
plt.figure(figsize=(8, 6))
sns.boxplot(x='Status', y='Life expectancy', data=combined_data, color='lightblue')
plt.title('Life Expectancy by Development Status')
plt.grid(axis='y', alpha=0.3)
plt.savefig('images/life_expectancy_by_status.png')
plt.close()

# 4. Bar chart for top and bottom countries
plt.figure(figsize=(12, 8))
countries = pd.concat([top_countries, bottom_countries])
countries = countries.sort_values('Life expectancy')

plt.barh(countries['Country'], countries['Life expectancy'], color='lightblue')
plt.xlabel('Life Expectancy (Years)')
plt.title('Countries with Highest and Lowest Life Expectancy')
plt.grid(axis='x', alpha=0.3)

# Add AQI values as text
for i, (_, row) in enumerate(countries.iterrows()):
    plt.text(row['Life expectancy'] + 0.5, i, f"AQI: {row['Average AQI']:.1f}")

plt.tight_layout()
plt.savefig('images/top_bottom_countries.png')
plt.close()

# 5. Correlation heatmap
plt.figure(figsize=(10, 8))
important_columns = ['Life expectancy', 'Average AQI', 'Adult Mortality', 'GDP', 'Schooling']
valid_columns = [col for col in important_columns if col in combined_data.columns]
correlation_matrix = combined_data[valid_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='Blues')
plt.title('Correlation Between Variables')
plt.tight_layout()
plt.savefig('images/correlation_heatmap.png')
plt.close()

# Statistical Analysis
# Test 1: Difference between low-risk and high-risk countries
low_risk = combined_data[combined_data['AQI Risk Category'] == 'Low Risk']['Life expectancy']
high_risk = combined_data[combined_data['AQI Risk Category'] == 'High Risk']['Life expectancy']

print("\nComparison between low-risk and high-risk countries:")
print(f"Low-risk average: {low_risk.mean():.2f} years")
print(f"High-risk average: {high_risk.mean():.2f} years")
print(f"Difference: {low_risk.mean() - high_risk.mean():.2f} years")

t_stat, p_value = stats.ttest_ind(low_risk, high_risk, equal_var=False)
print(f"p-value: {p_value:.4f}")
print("This difference is statistically significant" if p_value < 0.05 else "This difference is not statistically significant")

# Test 2: Difference between developed and developing countries
developed = combined_data[combined_data['Status'] == 'Developed']['Life expectancy']
developing = combined_data[combined_data['Status'] == 'Developing']['Life expectancy']

print("\nComparison between developed and developing countries:")
print(f"Developed average: {developed.mean():.2f} years")
print(f"Developing average: {developing.mean():.2f} years")
print(f"Difference: {developed.mean() - developing.mean():.2f} years")

t_stat, p_value = stats.ttest_ind(developed, developing, equal_var=False)
print(f"p-value: {p_value:.4f}")
print("This difference is statistically significant" if p_value < 0.05 else "This difference is not statistically significant")

# Test 3: Significance of correlation
correlation, p_value = stats.pearsonr(combined_data['Life expectancy'], combined_data['Average AQI'])
print(f"\nCorrelation significance test:")
print(f"Correlation: {correlation:.4f}, p-value: {p_value:.4f}")
print("This correlation is statistically significant" if p_value < 0.05 else "This correlation is not statistically significant")

print("\nVisualization files have been saved to the 'images' folder")