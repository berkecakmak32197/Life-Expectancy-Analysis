# Life Expectancy and Air Quality Analysis
## DSA 210 Introduction to Data Science Project

## Project Overview
This comprehensive data science project investigates the relationship between air quality and life expectancy across different countries, enhanced with additional environmental and socioeconomic factors. Through systematic data integration, feature engineering, and machine learning analysis, the project demonstrates how environmental conditions impact human longevity.

## Achievement
Integrated multiple data sources and improved model performance by 1.2% through careful feature engineering, proving that environmental factors significantly influence life expectancy predictions.

## Primary Goals
- **Quantify Environmental Impact**: Measure how air quality affects life expectancy using statistical analysis
- **Multi-Source Data Integration**: Combine life expectancy, air quality, CO2 emissions, and World Development Indicators
- **Advanced Feature Engineering**: Create composite features that capture complex relationships
- **Predictive Modeling**: Build and compare machine learning models for life expectancy prediction

## Technical Objectives
- Apply data preprocessing, statistical analysis, and visualization techniques
- Implement correlation analysis and hypothesis testing
- Develop and evaluate machine learning models
- Demonstrate redundancy detection and feature selection methods

## Public Health Significance
Air pollution affects billions globally and is a leading environmental health risk. Understanding its quantitative impact on life expectancy provides evidence for policy interventions.

## Data Science Application
This project applies comprehensive data science methodologies to a real-world problem, demonstrating skills in data integration, feature engineering, and predictive modeling.

## Datasets Used
Life Expectancy Data.csv          # WHO life expectancy data (2000-2015)
data_date.csv                     # Air quality index data (2022-2025)  
GCB2022v27_MtCO2_flat.csv       # Global CO2 emissions data
world-development-indicators.csv  # World Bank development indicators

## Core Analysis Script
  - Data loading and preprocessing
  - Multi-source data integration
  - Feature engineering and creation
  - Statistical analysis and hypothesis testing
  - Machine learning model development
  - Visualization generation

## Dataset Details

## Life Expectancy Data (WHO)
- **Source**: World Health Organization
- **Coverage**: 2000-2015, 183 countries
- **Key Variables**: Life expectancy, adult mortality, GDP, schooling, healthcare expenditure
- **Records**: 2,938 country-year observations

## Air Quality Data
- **Coverage**: 2022-2025, global AQI measurements
- **Key Variables**: AQI values, air quality status classifications
- **Records**: 18,227 measurements
- **Categories**: Good, Moderate, Unhealthy classifications

## CO2 Emissions Data
- **Source**: Global Carbon Budget 2022
- **Coverage**: Global CO2 emissions by country
- **Key Variables**: Per capita emissions, total emissions
- **Records**: 63,104 country-year observations

## World Development Indicators
- **Source**: World Bank
- **Coverage**: 56 development indicators
- **Key Indicators**: 
  - Electric power consumption (kWh per capita)
  - Births attended by skilled health staff (%)
  - Forest area (sq. km)
  - Urban population (% of total)
- **Records**: 1,160 indicator-country combinations

## Methodology

## Data Integration Process
1. **Primary Merge**: Life expectancy (2015) + Air quality (averaged)
2. **CO2 Integration**: Added per capita and total emissions
3. **WDI Enhancement**: Integrated 28 useful World Bank indicators
4. **Final Dataset**: 108 countries × 60 features

## Features
1. **Environmental_Impact_Score**: Combined CO2 and AQI impact
2. **Health_Investment_Index**: Healthcare spending and economic capacity  
3. **Basic_Services_Access**: Infrastructure access composite
4. **Development_Score**: Overall country development level
5. **Health_Risk_Factor**: Combined health and environmental risks

## Statistical Analysis
- **Correlation Analysis**: Pearson correlations with significance testing
- **Group Comparisons**: t-tests between developed/developing countries
- **Risk Categorization**: AQI-based risk level analysis
- **Redundancy Detection**: Feature correlation analysis (>0.8 threshold)

## Machine Learning
- **Model Type**: Linear Regression
- **Training Split**: 80/20 train-test split
- **Evaluation Metrics**: R² score, Mean Squared Error
- **Feature Selection**: Redundancy-based feature elimination

## Statistical Results
- **Air Quality Impact**: -0.31 correlation between AQI and life expectancy (*p* < 0.01)
- **Development Gap**: 9.61 years difference between developed and developing countries
- **Risk Categories**: 6.63 years difference between low-risk and high-risk air quality

## Model Performance
- **Enhanced Model R²**: 0.697
- **Basic Model R²**: 0.688  
- **Improvement**: +1.2% performance gain
- **Feature Importance**: Schooling (1.37), Development Status (0.44)

## Data Integration
- **Dataset Expansion**: From 29 to 60 features
- **Multi-Source Integration**: 4 major datasets successfully merged
- **Quality Indicators**: 28 World Bank indicators integrated
- **Coverage**: 108 countries with complete data


## Analysis Plots
1. **Life Expectancy vs AQI Scatter**: Shows negative correlation with trend line
2. **Risk Category Comparison**: Bar chart of average life expectancy by air quality risk
3. **Development Status Box Plot**: Comparison between developed/developing countries
4. **Top/Bottom Countries**: Horizontal bar chart of extreme cases
5. **Correlation Heatmap**: Feature correlation matrix with new engineered features
6. **Environmental Impact Analysis**: Environmental score vs life expectancy relationship
7. **ML Model Results**: Actual vs predicted values with feature importance


## Technical Implementation
## Libraries Used
pandas              # Data manipulation and analysis
numpy               # Numerical computations  
matplotlib.pyplot   # Basic plotting
seaborn            # Statistical visualizations
scipy.stats        # Statistical tests
sklearn            # Machine learning models and metrics
os                 # File system operations

## Functions
- **Data Loading**: Multi-format CSV handling with error management
- **Data Cleaning**: Missing value imputation and outlier detection
- **Feature Creation**: Mathematical combinations of existing variables
- **Statistical Testing**: Automated significance testing
- **Visualization**: Automated plot generation with customization

## Technical Achievements
- **Data Integration Mastery**: Successfully merged 4 disparate datasets
- **Feature Engineering**: Created meaningful composite indicators
- **Model Improvement**: Enhanced prediction accuracy through careful feature selection
- **Statistical Validation**: All major findings statistically significant

## Educational Value
- **Complete Data Science Pipeline**: From raw data to final insights
- **Real-World Application**: Applied techniques to meaningful public health problem
- **Advanced Techniques**: Feature engineering, redundancy detection, multi-source integration
- **Reproducible Research**: Well-documented, executable analysis

## Practical Insights
- **Environmental Health Impact**: Quantified relationship between air quality and longevity
- **Development Disparities**: Highlighted health inequalities between country types
- **Policy Implications**: Evidence for environmental and health interventions

Prerequisites:
pip install pandas numpy matplotlib seaborn scipy scikit-learn

Execution:
python life_expectancy_air_quality.py

Metric | Value | Significance |
AQI-Life Expectancy Correlation | -0.31 | *p* < 0.01 |
Developed vs Developing Gap | 9.61 years | *p* < 0.001 |
Low vs High Air Quality Risk | 6.63 years | *p* < 0.01 |
Enhanced Model R² | 0.697 | +1.2% improvement |
Dataset Features | 60 | 4 sources integrated |

- **Data Preprocessing**: Missing value handling, outlier detection
- **Statistical Analysis**: Correlation analysis, hypothesis testing  
- **Data Visualization**: Multiple plot types, statistical graphics
- **Machine Learning**: Supervised learning, model evaluation
- **Feature Engineering**: Creating meaningful derived variables
- **Research Methodology**: Systematic approach, reproducible results