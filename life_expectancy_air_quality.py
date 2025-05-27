# Life Expectancy and Air Quality Analysis
# DSA 210 Introduction to Data Science Project

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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


# DATA ENRICHMENT SECTION - Adding CO2 and World Development Data

print("\n")
print("\n")
print("DATA ENRICHMENT WITH ADDITIONAL DATASETS")
print("\n")
print("\n")

# Load CO2 emissions data
print("Loading CO2 emissions data...")
try:
    co2_data = pd.read_csv('GCB2022v27_MtCO2_flat.csv')
    print(f"Loaded CO2 data: {co2_data.shape[0]} rows × {co2_data.shape[1]} columns")
    
    # Filter for 2015 data and get per capita emissions
    co2_2015 = co2_data[co2_data['Year'] == 2015][['Country', 'Per Capita', 'Total']].copy()
    co2_2015 = co2_2015.rename(columns={
        'Per Capita': 'CO2_Per_Capita',
        'Total': 'CO2_Total_Emissions'
    })
    print(f"CO2 data for 2015: {len(co2_2015)} countries")
    
except Exception as e:
    print(f"Could not load CO2 data: {e}")
    # Create sample CO2 data if file not found
    co2_2015 = pd.DataFrame({
        'Country': combined_data['Country'].tolist(),
        'CO2_Per_Capita': np.random.uniform(1, 15, len(combined_data)),
        'CO2_Total_Emissions': np.random.uniform(50, 1000, len(combined_data))
    })
    print("Using sample CO2 data for demonstration")

# Load World Development Indicators
print("\nLoading World Development Indicators...")
try:
    wdi_data = pd.read_csv('world-development-indicators.csv')
    print(f"Loaded WDI data: {wdi_data.shape[0]} rows x {wdi_data.shape[1]} columns")
    
    print("Available indicators in WDI data:")
    available_series = wdi_data['Series Name'].unique()
    print(f"Total indicators: {len(available_series)}")

    # Show some examples
    for i, indicator in enumerate(available_series[:10]):
        print(f"  {i+1}. {indicator}")

    useful_patterns = [
        'urban', 'population', 'electric', 'sanitation', 'water', 
        'literacy', 'forest', 'health', 'expenditure', 'gdp',
        'mortality', 'birth', 'death', 'life expectancy'
    ]

    available_indicators = []
    for indicator in available_series:
        if not isinstance(indicator, str):
            continue
        indicator_lower = indicator.lower()
        for pattern in useful_patterns:
            if pattern in indicator_lower and indicator not in available_indicators:
                available_indicators.append(indicator)
                break

    print(f"\nFound {len(available_indicators)} potentially useful indicators:")
    for indicator in available_indicators[:10]: 
        print(f"  - {indicator}")
    useful_indicators = available_indicators[:7]  # Take first 7 useful ones
    
    if available_indicators:
        print(f"Found {len(available_indicators)} useful indicators in WDI data")
        
        # Filter and pivot the data
        wdi_filtered = wdi_data[
            wdi_data['Series Name'].isin(available_indicators) & 
            (~wdi_data['2015 [YR2015]'].isna())
        ][['Country Name', 'Series Name', '2015 [YR2015]']]
        
        wdi_filtered['2015 [YR2015]'] = pd.to_numeric(wdi_filtered['2015 [YR2015]'], errors='coerce')
        
        # Pivot to get indicators as columns
        wdi_pivot = wdi_filtered.pivot(index='Country Name', columns='Series Name', values='2015 [YR2015]').reset_index()
        wdi_pivot.columns.name = None
        wdi_pivot = wdi_pivot.rename(columns={'Country Name': 'Country'})
        
        column_mapping = {
            'Urban population (% of total population)': 'Urban_Population_Pct',
            'Access to electricity (% of population)': 'Electricity_Access_Pct',
            'Improved sanitation facilities (% of population with access)': 'Sanitation_Access_Pct',
            'Improved water source (% of population with access)': 'Water_Access_Pct',
            'Literacy rate, adult total (% of people ages 15 and above)': 'Adult_Literacy_Pct',
            'Forest area (% of land area)': 'Forest_Area_Pct',
            'Health expenditure, total (% of GDP)': 'Health_Expenditure_GDP_Pct'
        }
        
        existing_columns = {k: v for k, v in column_mapping.items() if k in wdi_pivot.columns}
        wdi_pivot = wdi_pivot.rename(columns=existing_columns)
        
        print(f"WDI data processed: {wdi_pivot.shape[0]} countries")
    else:
        print("No useful indicators found in WDI data, creating sample data")
        wdi_pivot = pd.DataFrame({
            'Country': combined_data['Country'].tolist(),
            'Urban_Population_Pct': np.random.uniform(30, 95, len(combined_data)),
            'Forest_Area_Pct': np.random.uniform(5, 70, len(combined_data))
        })
        
except Exception as e:
    print(f"Could not load WDI data: {e}")
    # Create minimal sample data
    wdi_pivot = pd.DataFrame({
        'Country': combined_data['Country'].tolist(),
        'Urban_Population_Pct': np.random.uniform(30, 95, len(combined_data)),
        'Forest_Area_Pct': np.random.uniform(5, 70, len(combined_data))
    })
    print("Using sample WDI data for demonstration")

# Merge all datasets
print("\nMerging all datasets...")
enriched_data = combined_data.copy()

# Merge CO2 data
enriched_data = pd.merge(enriched_data, co2_2015, on='Country', how='left')
co2_merged = enriched_data.dropna(subset=['CO2_Per_Capita']).shape[0]
print(f"Successfully merged CO2 data for {co2_merged} countries")

# Merge WDI data
enriched_data = pd.merge(enriched_data, wdi_pivot, on='Country', how='left')
wdi_merged = enriched_data.dropna(subset=['Urban_Population_Pct']).shape[0] if 'Urban_Population_Pct' in enriched_data.columns else 0
print(f"Successfully merged WDI data for {wdi_merged} countries")

print(f"Final enriched dataset: {enriched_data.shape[0]} countries x {enriched_data.shape[1]} features")

# CREATE FEATURES
print("\n")
print("\n")
print("CREATING FEATURES")
print("\n")
print("\n")

# Feature 1: Environmental Impact Score (combine CO2 and AQI)
enriched_data['Environmental_Impact_Score'] = (
    enriched_data['CO2_Per_Capita'].fillna(enriched_data['CO2_Per_Capita'].mean()) * 0.6 +
    enriched_data['Average AQI'] * 0.4
)

# Feature 2: Health Investment Index (combine health spending indicators)
enriched_data['Health_Investment_Index'] = (
    enriched_data['Total expenditure'].fillna(enriched_data['Total expenditure'].mean()) * 0.7 +
    enriched_data['GDP'] / 10000 * 0.3  # Normalize GDP contribution
)

# Feature 3: Basic Services Access (combine water, sanitation, electricity if available)
if 'Water_Access_Pct' in enriched_data.columns:
    enriched_data['Basic_Services_Access'] = (
        enriched_data['Water_Access_Pct'].fillna(80) * 0.4 +
        enriched_data.get('Sanitation_Access_Pct', pd.Series([75]*len(enriched_data))).fillna(75) * 0.3 +
        enriched_data.get('Electricity_Access_Pct', pd.Series([85]*len(enriched_data))).fillna(85) * 0.3
    )
else:
    # Create based on development status
    enriched_data['Basic_Services_Access'] = np.where(
        enriched_data['Status'] == 'Developed', 95, 70
    ) + np.random.uniform(-10, 10, len(enriched_data))

# Feature 4: Development Score (combine multiple development indicators)
enriched_data['Development_Score'] = (
    (enriched_data['GDP'] / enriched_data['GDP'].max() * 100) * 0.3 +  # Normalized GDP
    enriched_data['Schooling'].fillna(enriched_data['Schooling'].mean()) * 0.4 +
    enriched_data.get('Urban_Population_Pct', pd.Series([60]*len(enriched_data))).fillna(60) * 0.3
)

# Feature 5: Health Risk Factor (combine mortality and environmental risks)
enriched_data['Health_Risk_Factor'] = (
    enriched_data['Adult Mortality'] * 0.4 +
    enriched_data['Environmental_Impact_Score'] * 0.3 +
    (100 - enriched_data['Basic_Services_Access']) * 0.3  # Higher risk when less access
)

print("Created 5 new engineered features:")
print("1. Environmental_Impact_Score - Combined CO2 and air quality impact")
print("2. Health_Investment_Index - Healthcare spending and economic capacity")
print("3. Basic_Services_Access - Access to basic infrastructure")
print("4. Development_Score - Overall country development level")
print("5. Health_Risk_Factor - Combined health and environmental risks")

# Show correlations with life expectancy
print("\n")
print("\n")
print("CORRELATION WITH LIFE EXPECTANCY")
print("\n")
print("\n")

new_features = ['Environmental_Impact_Score', 'Health_Investment_Index', 'Basic_Services_Access', 
                'Development_Score', 'Health_Risk_Factor']

correlations = []
for feature in new_features:
    corr = enriched_data['Life expectancy'].corr(enriched_data[feature])
    correlations.append((feature, corr))
    print(f"{feature}: {corr:.3f}")

# Update combined_data for the rest of the analysis
combined_data = enriched_data.copy()

print(f"\nData enrichment completed! Dataset now has {combined_data.shape[1]} features.")

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

# 5. Correlation heatmap with new features
plt.figure(figsize=(12, 10))
important_columns = ['Life expectancy', 'Average AQI', 'Adult Mortality', 'GDP', 'Schooling',
                    'Environmental_Impact_Score', 'Health_Investment_Index', 'Development_Score']
valid_columns = [col for col in important_columns if col in combined_data.columns]
correlation_matrix = combined_data[valid_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt='.3f')
plt.title('Correlation Matrix with New Features')
plt.tight_layout()
plt.savefig('images/correlation_heatmap.png')
plt.close()

# 6. New visualization: Environmental Impact vs Life Expectancy
plt.figure(figsize=(10, 6))
plt.scatter(combined_data['Environmental_Impact_Score'], combined_data['Life expectancy'], 
           color='green', alpha=0.7)
plt.title('Life Expectancy vs Environmental Impact Score')
plt.xlabel('Environmental Impact Score (Higher = More Environmental Damage)')
plt.ylabel('Life Expectancy (Years)')
plt.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(combined_data['Environmental_Impact_Score'], combined_data['Life expectancy'], 1)
p = np.poly1d(z)
plt.plot(combined_data['Environmental_Impact_Score'], p(combined_data['Environmental_Impact_Score']), "r--", alpha=0.7)
plt.savefig('images/environmental_impact_vs_life_expectancy.png')
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


# MACHINE LEARNING ANALYSIS with Enriched Features

print("\n" + "=" * 50)
print("MACHINE LEARNING ANALYSIS")
print("=" * 50)

# Check for missing values in features
print("\nChecking for missing values in features:")
enhanced_columns = ['Life expectancy', 'Average AQI', 'Adult Mortality', 'GDP', 'Schooling',
                   'Environmental_Impact_Score', 'Health_Investment_Index', 'Development_Score',
                   'Health_Risk_Factor', 'Basic_Services_Access']
available_columns = [col for col in enhanced_columns if col in combined_data.columns]
print(combined_data[available_columns].isnull().sum())

# Fill missing values
for column in available_columns:
    if combined_data[column].isnull().sum() > 0:
        avg_value = combined_data[column].mean()
        combined_data[column].fillna(avg_value, inplace=True)
        print(f"- Filled missing values in {column} with the average: {avg_value:.2f}")

# Prepare features for ML
combined_data['Is_Developed'] = (combined_data['Status'] == 'Developed').astype(int)

print("\nAnalyzing feature correlations to remove redundancy...")

feature_candidates = ['Average AQI', 'Adult Mortality', 'GDP', 'Schooling', 'Is_Developed',
                     'Environmental_Impact_Score', 'Health_Investment_Index', 'Development_Score', 'Health_Risk_Factor']

feature_corr = combined_data[feature_candidates].corr()

redundant_features = []
for i in range(len(feature_corr.columns)):
    for j in range(i+1, len(feature_corr.columns)):
        if abs(feature_corr.iloc[i, j]) > 0.8:
            redundant_features.append((feature_corr.columns[i], feature_corr.columns[j], feature_corr.iloc[i, j]))

if redundant_features:
    print("\nFound highly correlated feature pairs:")
    for feat1, feat2, corr in redundant_features:
        print(f"  {feat1} ↔ {feat2}: {corr:.3f}")

enhanced_features = [
    'Adult Mortality', 'Schooling', 'Is_Developed',
    'Health_Risk_Factor', 
    'Basic_Services_Access',  
]

print(f"\nUsing features (removed redundant ones): {enhanced_features}")

features = [f for f in enhanced_features if f in combined_data.columns]
print(f"\nUsing features for prediction: {features}")

# Prepare data
X = combined_data[features]
y = combined_data['Life expectancy']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nSplit data: {len(X_train)} training, {len(X_test)} testing countries")

# Train model
print("\nTraining linear regression model...")
enhanced_model = LinearRegression()
enhanced_model.fit(X_train, y_train)

# Make predictions
y_pred_enhanced = enhanced_model.predict(X_test)

# Evaluate model
mse_enhanced = mean_squared_error(y_test, y_pred_enhanced)
r2_enhanced = r2_score(y_test, y_pred_enhanced)

print(f"\nModel Results:")
print(f"- Mean Squared Error: {mse_enhanced:.2f}")
print(f"- R² Score: {r2_enhanced:.2f}")

# Feature importance
print("\nFeature Importance:")
for feature, coefficient in zip(features, enhanced_model.coef_):
    print(f"- {feature}: {coefficient:.4f}")

# Compare with original model (using basic features)
basic_features = ['Average AQI', 'Adult Mortality', 'GDP', 'Schooling', 'Is_Developed']
X_basic = combined_data[basic_features]
X_train_basic, X_test_basic, _, _ = train_test_split(X_basic, y, test_size=0.2, random_state=42)

basic_model = LinearRegression()
basic_model.fit(X_train_basic, y_train)
y_pred_basic = basic_model.predict(X_test_basic)
r2_basic = r2_score(y_test, y_pred_basic)

print(f"\nModel Comparison:")
print(f"\n- Basic Model R²: {r2_basic:.3f}")
print(f"\n- Enhanced Model R²: {r2_enhanced:.3f}")
improvement = ((r2_enhanced - r2_basic) / r2_basic * 100)
print(f"\n- Change: {improvement:+.1f}%")

if improvement > 0:
    print("\nNew model performs better!")
else:
    print("\nNew model performs worse - this suggests feature redundancy")
    print("\nThe new features may be too similar to existing ones")
    print("\nShould consider: fewer features, different combinations, or feature selection")

# Visualizations for model
plt.figure(figsize=(12, 5))

#  actual vs predicted
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_enhanced, color='green', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual Life Expectancy')
plt.ylabel('Predicted Life Expectancy')
plt.title(f'Model: Actual vs Predicted (R² = {r2_enhanced:.3f})')
plt.grid(True, alpha=0.3)

# Feature importance
plt.subplot(1, 2, 2)
abs_coefficients = [abs(coef) for coef in enhanced_model.coef_]
importance_df = pd.DataFrame({'Feature': features, 'Importance': abs_coefficients})
importance_df = importance_df.sort_values('Importance', ascending=True)

plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.title('Model Feature Importance')
plt.xlabel('Absolute Coefficient Value')

plt.tight_layout()
plt.savefig('images/_ml_results.png', dpi=300, bbox_inches='tight')
plt.close()

# Prediction analysis
print("\n Model Predictions for Different Scenarios:")
print("Using model to predict life expectancy...")

# Create scenarios with features
scenarios = [
    {"name": "Low Environmental Impact, High Development", 
     "Environmental_Impact_Score": 30, "Development_Score": 90},
    {"name": "High Environmental Impact, Low Development", 
     "Environmental_Impact_Score": 120, "Development_Score": 40},
    {"name": "Moderate Environmental Impact, High Health Investment", 
     "Environmental_Impact_Score": 70, "Health_Investment_Index": 85},
    {"name": "High Environmental Risk, Poor Health System", 
     "Environmental_Impact_Score": 100, "Health_Investment_Index": 25}
]

# Use average values for other features
avg_values = {}
for feature in features:
    if feature not in ['Environmental_Impact_Score', 'Development_Score', 'Health_Investment_Index']:
        avg_values[feature] = X[feature].mean()

for scenario in scenarios:
    test_data = avg_values.copy()
    test_data.update(scenario)
    
    # Remove name from test data
    scenario_name = test_data.pop('name', scenario['name'])
    
    # Ensure all features are present
    for feature in features:
        if feature not in test_data:
            test_data[feature] = X[feature].mean()
    
    # Create DataFrame with correct feature order
    test_df = pd.DataFrame([test_data])[features]
    prediction = enhanced_model.predict(test_df)[0]
    
    print(f"\n- {scenario_name}:")
    print(f"  Predicted Life Expectancy: {prediction:.2f} years")

print("\nMachine learning analysis completed!")
print("\nVisualizations saved to 'images' folder.")