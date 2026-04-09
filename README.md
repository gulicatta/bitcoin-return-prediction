# Bitcoin Return Prediction

Machine learning project on Bitcoin return prediction using:
* Linear Regression
* Random Forest
* Support Vector Regression (SVR)

## Project overview
The project compares two feature sets:
1. **Raw market variables:** open, close, high, low, volume
2. **Engineered predictors/factors:** Log-MarketCap, Amihud Illiquidity, rolling maximums, and historical log-returns.

The target is the 7-day forward return based on the closing price.

## Main findings
* **Factors superiority:** Factor-based models consistently performed better than raw-price models in terms of test RMSE.
* **Winning Model:** Linear Regression with engineered factors achieved the best out-of-sample test performance.
* **Beating the Market:** A simple long/cash trading strategy based on the Linear Regression (Factors) predictions **outperformed the Buy & Hold benchmark** over the selected test period (closing at ~€273k vs the ~€213k of Buy & Hold, starting from a €100k initial investment).
* **Overfitting in Non-Linear Models:** Random Forest and SVR struggled to beat the benchmark, likely due to their tendency to overfit the historical market noise, leading to very low market exposure during the test period.

## Repository structure
* `main.py`: main project script
* `results/`: saved figures (feature importance, strategy comparison)
* `README.md`: project description
* `requirements.txt`: Python dependencies

## Dataset
The dataset is not included in this repository to respect the assignment's data sharing rules. 
To run the project, place your `BTC_daily_historical.csv` dataset file inside a `data/` folder and update the file path in the script if needed.

## Note
The assignment text refers to 10 predictors, but the implemented version uses 11 engineered predictors (including the Amihud Illiquidity measure, `DAMIHUD`, to capture market friction).