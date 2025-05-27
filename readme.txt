# Life Expectancy and Air Quality Analysis

This project explores how air quality affects life expectancy across different countries. I wanted to see if countries with cleaner air tend to have longer lifespans, and if this relationship is statistically significant.

## Motivation

I chose this topic because:
- Air pollution is a big environmental health issue affecting lots of people
- I wanted to see if the data supports the idea that cleaner air leads to longer lives
- It combines public health and environmental data in an interesting way

## Data Sources

I used two main datasets:
1. **Life Expectancy Data** from WHO (covering 2000-2015)
   - Contains health and economic data for 193 countries
   - Includes life expectancy, mortality rates, disease prevalence, GDP, etc.

2. **Air Quality Data** (covering 2022-2025)
   - Contains Air Quality Index (AQI) measurements for 142 countries
   - Includes AQI values and air quality status categories (Good, Moderate, Unhealthy, etc.)

One challenge was that these datasets cover different time periods. To address this, I used the most recent life expectancy data (2015) and average air quality values for each country.

## How to Run This Analysis

### Requirements

- Python 3.6 or higher
- Required libraries: pandas, numpy, matplotlib, seaborn, scipy

### Installation

1. Clone this repository or download the files
2. Make sure both data files (`Life Expectancy Data.csv` and `data_date.csv`) are in the same folder as the script
3. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

### Running the Analysis

Simply run the main Python script:
```
python life_expectancy_air_quality_analysis.py
```

The script will:
- Load and explore both datasets
- Clean and prepare the data
- Analyze relationships between air quality and life expectancy
- Perform hypothesis tests
- Create visualizations in an 'images' folder

## Key Findings

1. **Dataset Overview**:
   - Analyzed data from 108 countries with both life expectancy (2015) and air quality data
   - 32 developed countries and 76 developing countries

2. **Relationship between Air Quality and Life Expectancy**:
   - Correlation between Life Expectancy and Average AQI: -0.3097
   - Negative correlation means that as air pollution increases, life expectancy tends to decrease
   - This correlation is statistically significant (p-value: 0.0011)

3. **Life Expectancy by Air Quality Risk Category**:
   - Low Risk (AQI ≤ 50): 76.94 years
   - Moderate Risk (AQI 51-100): 73.15 years
   - High Risk (AQI > 100): 70.31 years
   - The difference between Low Risk and High Risk countries is statistically significant

4. **Life Expectancy by Development Status**:
   - Developed countries: 80.77 years
   - Developing countries: 71.16 years
   - Difference: 9.61 years
   - This difference is statistically significant

5. **Interesting Findings**:
   - Countries with cleaner air tend to have higher life expectancy
   - Developed countries generally have both better air quality and higher life expectancy
   - Some developing countries with poor air quality still manage to have relatively high life expectancy

## Limitations

- Time gap between datasets (life expectancy from 2015, air quality from recent years)
- Country-level analysis misses regional variations within countries
- Correlation doesn't necessarily mean causation - other factors may be involved
- Limited sample size (only ~100 countries with data in both datasets)

## Future Work

For future analysis, I'd like to:
- Get datasets where life expectancy and air quality are measured in the same time period
- Include more specific air pollutant measurements (PM2.5, NO2, etc.)
- Do regional or city-level analysis rather than country-level
- Look at how changes in air quality over time relate to changes in life expectancy
- Apply machine learning techniques to predict life expectancy based on air quality and other factors

## Files in this Project

- `life_expectancy_air_quality_analysis.py`: Main Python script for analysis
- `Life Expectancy Data.csv`: WHO life expectancy dataset
- `data_date.csv`: Air quality dataset
- `requirements.txt`: Required Python packages
- `images/`: Folder containing generated visualizations