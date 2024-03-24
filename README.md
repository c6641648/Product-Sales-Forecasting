### Quantitative Forecasting Methods for Product Sales Projection

In a dedicated pursuit of mastering time series analysis and forecasting techniques, a comprehensive exploration of quantitative forecasting methods was undertaken. This project delves into various methodologies applicable for projecting product sales, accompanied by an exhaustive elucidation of metrics crucial for evaluating forecast accuracy.

Objective:*
The primary objective of this endeavor was to employ diverse quantitative methods—ranging from Time Series Models to Causal Models—to predict the sales trajectory of products sourced from a dataset procured from Kaggle.

Featured Models:

1. Seasonal Naive Model
2. Holt-Winters Model (Triple Exponential Smoothing)
3. ARIMA and Seasonal ARIMA Models
4. Linear Regression Model

This repository's notebook meticulously documents the application of these models and provides in-depth insights into the evaluation metrics employed to gauge forecast performance.



**Product Sales Forecasting using Quantitative Methods**

**1. Introduction**
   - Brief overview of the project's objective and scope.
   - Mention the importance of forecasting sales for businesses.

**2. Dataset**
   - Description of the dataset used, including the source (Kaggle), number of stores, products, and time period covered (2013-2017).
   - Specifics about the data preprocessing steps, such as filtering for store 1 and item 1, creating date-related features, and splitting into train and test sets.

**3. Data Exploration**
   - Visualizations depicting weekly, monthly, and yearly sales distributions to understand sales trends.
   - Description of the observed trends and patterns in the data.

**4. Quantitative Methods for Sales Forecasting**
   - Explanation of the Seasonal Naive baseline model and its approach.
   - Introduction to Exponential Smoothing and its relevance to time series forecasting.
   - Brief overview of other models to be covered (Holt-Winters, ARIMA, Seasonal ARIMA, Linear Regression).

**5. Baseline Model: Seasonal Naive**
   - Details of the Seasonal Naive model, including how it predicts sales based on the previous year's data.
   - Performance evaluation metrics used, such as MAE, RMSE, and MAPE.
   - Visualization of forecasts and actual sales, along with errors.

**6. Time Series Decomposition Plot**
   - Explanation of the decomposition plot and its importance in selecting forecasting models.
   - Description of the observed components (trend, seasonality, error) and their properties (additive or multiplicative).

**7. Model Selection and Evaluation**
   - Discussion on how the properties observed in the decomposition plot guide the selection of Exponential Smoothing models.
   - Overview of other models considered and their suitability for the dataset.
   - Evaluation of model performance using relevant metrics and comparison with the Seasonal Naive baseline.

**8. Conclusion**
   - Summary of key findings from the analysis.
   - Recommendations for further steps or improvements in forecasting accuracy.
   - Closing remarks on the significance of quantitative methods in sales forecasting.




### Seasonal Autoregressive Integrated Moving-Average (SARIMA) Model

#### Step 1: Model Building
- **Initialization**: 
    - The SARIMA model is initialized with the SARIMAX function from the `statsmodels.tsa.statespace.sarimax` module.
    - `order=(6, 1, 0)` indicates the non-seasonal ARIMA parameters: `p=6` (number of autoregressive terms), `d=1` (order of differencing), and `q=0` (number of moving average terms).
    - `seasonal_order=(6, 1, 0, 7)` indicates the seasonal ARIMA parameters: `P=6` (number of seasonal autoregressive terms), `D=1` (order of seasonal differencing), `Q=0` (number of seasonal moving average terms), and `s=7` (seasonal period).

#### Step 2: Model Fitting
- The SARIMA model is fitted to the training data using the `fit()` function.
- The fitted model is then used to generate forecasts for the test period.

#### Step 3: Diagnostic Analysis
- **Residual Analysis**:
    - The residuals of the SARIMA model are extracted using the `resid` attribute of the fitted model.
    - Autocorrelation and partial autocorrelation plots of the residuals are generated to check for any remaining patterns or correlations.

#### Step 4: Evaluation
- **Forecast Visualization**:
    - The forecasts made by the SARIMA model are plotted alongside the actual sales data for both the training and test periods.
    - This visualization helps in assessing the accuracy of the forecasts.

- **Error Analysis**:
    - The errors (the differences between actual and predicted sales) are calculated and plotted to evaluate the performance of the SARIMA model.
    - Metrics such as Mean Absolute Percentage Error (MAPE) are calculated to quantify the accuracy of the forecasts.

#### Step 5: Result Summary
- The results of the SARIMA model evaluation, including total sales, total predicted sales, overall error, MAE, RMSE, and MAPE, are aggregated and presented in a dataframe (`result_df_sarima`).

#### Conclusion (Inference):
- The inference drawn from the SARIMA model evaluation is provided, comparing its performance against the baseline model and the Holt-Winters Triple Exponential Smoothing method.
- In this specific case, the SARIMA model achieved a MAPE of 23.7%, performing better than the baseline model but not surpassing the Holt-Winters method.


This final part of our project focuses on applying a supervised machine learning technique, specifically Linear Regression, for forecasting sales data. 

### Linear Regression for Time Series Forecasting

#### Step 1: Feature Engineering
- **Lag Features**: Lag features are created by shifting the sales data by different time periods (lags). This allows the model to capture temporal dependencies.
- **Rolling Window Statistics**: Rolling mean, maximum, and minimum features are calculated over a window of 7 days. These features provide information about the trend and variability of sales over time.

#### Step 2: Feature Selection and Model Building
- **Feature Selection**:
    - The correlation matrix between the features and the target variable (sales) is computed.
    - The top 5 features with the highest correlation scores with sales are selected using the SelectKBest method with F-regression as the scoring function.
    - Scatter plots are generated to visualize the linear relationship between the selected features and sales.

- **Model Fitting**:
    - Linear Regression model is initialized and fitted to the training data using the selected features.

#### Step 3: Model Evaluation and Predictions
- **Evaluation**:
    - The model is evaluated by making predictions on the test data.
    - The forecasts made by the Linear Regression model are plotted alongside the actual sales data for both the training and test periods.
    - Error analysis is performed by plotting the errors (the differences between actual and predicted sales) over time.

- **Result Summary**:
    - The results of the Linear Regression model evaluation, including total sales, total predicted sales, overall error, MAE, RMSE, and MAPE, are aggregated and presented in a dataframe (`result_df_lr`).

#### Conclusion (Inference):
- The inference drawn from the Linear Regression model evaluation is provided, indicating that the model captures both upward and downward movements in the sales data.
- The performance of the Linear Regression model, with a MAPE of 19.07%, is reported to be better than the Holt-Winters model, which was previously the best-performing model.
- The conclusion suggests that for this dataset, a regression model is more suitable for forecasting sales compared to time-series models. The success of the Linear Regression model can be attributed to the dataset's highly seasonal nature and linear trend, which align well with the assumptions of regression models.

By systematically applying feature engineering, model building, and evaluation techniques, the project demonstrates the effectiveness of Linear Regression for time series forecasting in the context of sales data.