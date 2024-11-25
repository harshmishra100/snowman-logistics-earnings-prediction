# Predicting Q2 Earnings for Snowman Logistics  

## Objective  
This project predicts Snowman Logistics' Q2 earnings by utilizing advanced data science techniques, including:  
- **SARIMAX**: Time series modeling with external factors.  
- **Multiple Linear Regression (MLR)**: Feature-driven earnings estimation.  
- **NLP Sentiment Analysis**: Assessing market sentiment from industry news and reports.  

## Data Collection  
- **Historical Financial Data**: Quarterly earnings, revenue, expenses, EPS.  
- **External Factors**: Fuel costs, inflation rates, competitor performance metrics.  
- **Market Sentiment Data**: Extracted from news articles and sector reports.  

## Methodology  
1. **Data Preprocessing**:  
   - Handled missing values through interpolation and imputation.  
   - Normalized financial metrics to improve model performance.  
2. **Modeling**:  
   - **SARIMAX**: Captured seasonal patterns and external economic factors.  
   - **MLR**: Quantified the influence of financial and economic indicators.  
   - **NLP Sentiment Analysis**: Derived sentiment scores for market perception.  
3. **Evaluation**:  
   - Metrics used: RMSE, MAE, and R-squared for quantitative models.  
   - Comparison of model predictions ensured robustness.  

## Results  
- **Predicted Q2 Earnings**:  
  - SARIMAX: **22.84 million** (with a tight confidence interval).  
  - MLR: **22.10 million**, highlighting feature-driven insights.  
- **Sentiment Analysis**: Neutral sentiment indicates a balanced outlook.  

## Key Insights  
- The consistent predictions from SARIMAX and MLR highlight stable earnings.  
- Neutral sentiment reflects low volatility and a steady financial outlook.  
- Suggestion: **Hold recommendation** for low-risk investment.  

## Visualizations and Code  
The repository includes:  
- Visualizations for earnings trends and feature correlations.  
- Code for SARIMAX, MLR, and NLP implementation.  

## Installation  
1. Clone this repository:  
   ```bash
   git clone https://github.com/your-username/snowman-logistics-earnings-prediction.git
