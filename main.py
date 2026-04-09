import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance

# ==========================================
# 1. READ DATA & PROPERTIES (Task 1)
# ==========================================
filename = 'data/BTC_daily_historical.csv'

try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    raise FileNotFoundError(
        "Dataset not found. Place 'btchistorical.csv' inside the 'data/' folder."
    )

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

print("--- Descriptive Table (Task 1) ---")
print(df.describe())

# ==========================================
# 2. CONSTRUCT 11 PREDICTORS (Task 2)
# ==========================================
p = df['close']
mcap = df['marketCap']
vol = df['volume']

df['MCAP'] = np.log(mcap)
df['PRC'] = np.log(p)
df['MAXDPRC'] = p.rolling(window=7).max()

df['r1_0'] = p - p.shift(6)
df['r2_0'] = p - p.shift(13)
df['r3_0'] = p - p.shift(20)
df['r4_0'] = p - p.shift(27)
df['r4_1'] = p.shift(6) - p.shift(27)

df['PRCVOL'] = np.log(vol.rolling(window=7).mean() + 1)
df['STDPRCVOL'] = np.log(vol.rolling(window=7).std() + 1)

daily_ret = (p - p.shift(1)).abs() / p.shift(1)
df['DAMIHUD'] = (daily_ret.rolling(window=7).mean()) / (df['PRCVOL'] + 1e-9)

predictors = ['MCAP', 'PRC', 'MAXDPRC', 'r1_0', 'r2_0', 'r3_0',
              'r4_0', 'r4_1', 'PRCVOL', 'STDPRCVOL', 'DAMIHUD']

print("\n--- Descriptive Table (Task 2) ---")
print(df[predictors].describe().transpose()[['mean', 'std', 'min', 'max']])

# ==========================================
# 3. TARGET CALCULATION (Task 3)
# ==========================================
df['Target_R1'] = (p.shift(-6) - p) / p

print("\n--- Target Distribution (Task 3) ---")
print(df['Target_R1'].describe())

# ==========================================
# 4 & 5. TRAINING, VALIDATION, TESTING (Tasks 4 & 5)
# ==========================================
df_ml = df.dropna().copy()

test_mask = (df_ml['date'] >= '2024-01-01') & (df_ml['date'] <= '2025-04-30')
val_mask = (df_ml['date'] >= '2023-01-01') & (df_ml['date'] < '2024-01-01')
train_mask = (df_ml['date'] < '2023-01-01')

features_prices = ['open', 'close', 'high', 'low', 'volume']
features_factors = predictors


def run_ml(f_cols, label):
    local_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "SVR": SVR(kernel='rbf')
    }
    local_scaler = StandardScaler()

    X_train = local_scaler.fit_transform(df_ml.loc[train_mask, f_cols])
    X_val = local_scaler.transform(df_ml.loc[val_mask, f_cols])
    X_test = local_scaler.transform(df_ml.loc[test_mask, f_cols])

    y_train = df_ml.loc[train_mask, 'Target_R1']
    y_val = df_ml.loc[val_mask, 'Target_R1']
    y_test = df_ml.loc[test_mask, 'Target_R1']

    print(f"\n--- {label} Results ---")
    preds_dict = {}
    fitted_models = {}
    for name, model in local_models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model

        val_preds = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))

        test_preds = model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

        print(f"{name} -> Val RMSE: {val_rmse:.5f} | Test RMSE: {test_rmse:.5f}")
        preds_dict[name] = test_preds

    return preds_dict, fitted_models, local_scaler


prices_preds, _, _ = run_ml(features_prices, "Raw Prices (Task 4)")
factors_preds, factors_models, factors_scaler = run_ml(
    features_factors, "Factors (Task 5)")

# ==========================================
# 6. SIGNIFICANT DRIVERS (Task 6)
# ==========================================
rf = factors_models['Random Forest']
rf_importances = pd.Series(rf.feature_importances_,
                           index=features_factors).sort_values(ascending=False)

lr = factors_models['Linear Regression']
lr_importances = pd.Series(
    np.abs(lr.coef_), index=features_factors).sort_values(ascending=False)

svr = factors_models['SVR']
X_test_factors = factors_scaler.transform(
    df_ml.loc[test_mask, features_factors])
perm_importance = permutation_importance(svr,
                                         X_test_factors,
                                         df_ml.loc[test_mask, 'Target_R1'],
                                         n_repeats=10,
                                         random_state=42)
svr_importances = pd.Series(perm_importance.importances_mean,
                            index=features_factors).sort_values(ascending=False)

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

rf_importances.head(10).plot(kind='barh', color='green', ax=axes[0])
axes[0].set_title('Random Forest Importance (MDI)')
axes[0].invert_yaxis()
axes[0].set_xlabel('Importance Score')

lr_importances.head(10).plot(kind='barh', color='blue', ax=axes[1])
axes[1].set_title('Linear Regression Importance (Abs Coef)')
axes[1].invert_yaxis()
axes[1].set_xlabel('Absolute Coefficient')

svr_importances.head(10).plot(kind='barh', color='red', ax=axes[2])
axes[2].set_title('SVR Importance (Permutation)')
axes[2].invert_yaxis()
axes[2].set_xlabel('Mean Accuracy Decrease')

plt.tight_layout()
plt.savefig('drivers_importance.png')
plt.show()

print("\n--- TOP 3 DRIVERS PER MODEL ---")
print(f"Random Forest:     {rf_importances.index[:3].tolist()}")
print(f"Linear Regression: {lr_importances.index[:3].tolist()}")
print(f"SVR (RBF):         {svr_importances.index[:3].tolist()}")

# ==========================================
# 7. TRADING STRATEGY (Task 7)
# ==========================================
test_df = df_ml[test_mask].copy()
initial_inv = 100000.0


def backtest_strategy(prices, predictions, initial_capital):
    signals = pd.Series(predictions, index=prices.index).shift(1).fillna(0)
    btc_units = initial_capital / prices.iloc[0]
    cash = 0.0
    in_market = True
    portfolio_history = []

    for i in range(len(prices)):
        price = prices.iloc[i]
        pred = signals.iloc[i]

        if pred < 0 and in_market:
            cash = btc_units * price
            btc_units = 0.0
            in_market = False

        elif pred > 0 and not in_market:
            btc_units = cash / price
            cash = 0.0
            in_market = True

        portfolio_history.append(cash + (btc_units * price))

    return portfolio_history


buy_and_hold_hist = (initial_inv / test_df['close'].iloc[0]) * test_df['close']

print(f"\n--- Trading Task 7 Results ---")
print(f"Buy & Hold Final Value: €{buy_and_hold_hist.iloc[-1]:,.2f}")

strategies_factors = {}
for model_name, preds in factors_preds.items():
    hist = backtest_strategy(test_df['close'], preds, initial_inv)
    strategies_factors[model_name] = hist
    print(f"Factors - {model_name} Final Value: €{hist[-1]:,.2f}")

strategies_prices = {}
for model_name, preds in prices_preds.items():
    hist = backtest_strategy(test_df['close'], preds, initial_inv)
    strategies_prices[model_name] = hist
    print(f"Raw Prices - {model_name} Final Value: €{hist[-1]:,.2f}")

plt.figure(figsize=(14, 8))
plt.plot(test_df['date'], buy_and_hold_hist, label='Buy & Hold',
         color='black', linestyle='--', linewidth=2, alpha=0.8)

colors = {'Linear Regression': 'blue', 'Random Forest': 'green', 'SVR': 'red'}

for model_name, hist in strategies_factors.items():
    plt.plot(test_df['date'], hist, label=f'{model_name} (Factors)',
             color=colors.get(model_name, 'orange'), linestyle='-')

for model_name, hist in strategies_prices.items():
    plt.plot(test_df['date'], hist, label=f'{model_name} (Raw Prices)',
             color=colors.get(model_name, 'orange'), linestyle=':')

plt.title('Task 7: Trading Strategy Comparison (Factors vs Raw Prices vs Buy & Hold)', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Portfolio Value (€)')
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('trading_comparison.png')
plt.show()


def exposure_rate(preds):
    signals = pd.Series(preds).shift(1).fillna(0)
    in_market = True
    pos = []
    for p in signals:
        if (p < 0) and in_market:
            in_market = False
        elif (p > 0) and (not in_market):
            in_market = True
        pos.append(1 if in_market else 0)
    return np.mean(pos)


exposure = pd.DataFrame({
    "Model (Task 5)": ["LR", "RF", "SVR"],
    "Exposure (share of days invested)": [
        exposure_rate(factors_preds["Linear Regression"]),
        exposure_rate(factors_preds["Random Forest"]),
        exposure_rate(factors_preds["SVR"])
    ]
})
exposure["Exposure (%)"] = (
    exposure["Exposure (share of days invested)"] * 100).round(2)

print("\n--- Exposure Rate (Task 7 Add-on) ---")
print(exposure)

y_test_vals = df_ml.loc[test_mask, "Target_R1"].values


def dir_acc(y_true, y_pred):
    return (np.sign(y_true) == np.sign(y_pred)).mean()


dir_table = pd.DataFrame({
    "Model (Task 5)": ["Linear Regression", "Random Forest", "SVR"],
    "Directional Accuracy": [
        dir_acc(y_test_vals, factors_preds["Linear Regression"]),
        dir_acc(y_test_vals, factors_preds["Random Forest"]),
        dir_acc(y_test_vals, factors_preds["SVR"])
    ]
})
dir_table["Directional Accuracy (%)"] = (
    dir_table["Directional Accuracy"] * 100).round(2)

print("\n--- Directional Accuracy (Task 7 Add-on) ---")
print(dir_table)
