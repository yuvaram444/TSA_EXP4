# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 11-05-26



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ── 1. Load & Prepare Data ────────────────────────────────────────────────────
STOCK    = 'HDFCBANK.NS'
CSV_PATH = 'nifty50_2000_2025.csv'   # keep CSV in same folder as notebook

df = pd.read_csv(CSV_PATH)
df['Date'] = pd.to_datetime(df['Date'])

stock_df = df[df['Stock'] == STOCK].sort_values('Date').copy()
stock_df['YearMonth'] = stock_df['Date'].dt.to_period('M')

# Monthly average Close price  (replaces 'money' column in original)
monthly = stock_df.groupby('YearMonth')['Close'].mean().reset_index()
monthly.index = monthly['YearMonth'].dt.to_timestamp()

X = monthly['Close']

print(f"Stock       : {STOCK}")
print(f"Data points : {len(X)}  (monthly avg close, 2000–2024)\n")

# ── 2. Plot Original Series ───────────────────────────────────────────────────
plt.rcParams['figure.figsize'] = [12, 6]

plt.plot(X)
plt.title(f'Original Monthly Avg Close Price — {STOCK}')
plt.xlabel('Date')
plt.ylabel('Close Price (₹)')
plt.show()

# ── 3. ACF & PACF of Original Data ───────────────────────────────────────────
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plot_acf(X, lags=40, ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=40, ax=plt.gca())
plt.title('Original Data PACF')

plt.tight_layout()
plt.show()

# ── 4. Fit ARMA(1,1) ──────────────────────────────────────────────────────────
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()

print("\nARMA(1,1) Model Summary")
print(arma11_model.summary())

# Extract fitted coefficients
phi1_arma11   = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1,  theta1_arma11])

# Simulate ARMA(1,1) process using fitted coefficients
N = 1000
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.figure(figsize=(12, 6))
plt.plot(ARMA_1)
plt.title(f'Simulated ARMA(1,1) — {STOCK} Close Price')
plt.xlabel('Samples')
plt.ylabel('Values')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_1)
plt.title('ACF of ARMA(1,1)')
plt.show()

plot_pacf(ARMA_1)
plt.title('PACF of ARMA(1,1)')
plt.show()

# ── 5. Fit ARMA(2,2) ──────────────────────────────────────────────────────────
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()

print("\nARMA(2,2) Model Summary")
print(arma22_model.summary())

# Extract fitted coefficients
phi1_arma22   = arma22_model.params['ar.L1']
phi2_arma22   = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1,  theta1_arma22, theta2_arma22])

# Simulate ARMA(2,2) process using fitted coefficients
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N * 10)

plt.figure(figsize=(12, 6))
plt.plot(ARMA_2)
plt.title(f'Simulated ARMA(2,2) — {STOCK} Close Price')
plt.xlabel('Samples')
plt.ylabel('Values')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_2)
plt.title('ACF of ARMA(2,2)')
plt.show()

plot_pacf(ARMA_2)
plt.title('PACF of ARMA(2,2)')
plt.show()
```

### OUTPUT:

ORIGINAL MONTHLY AVG CLOSE PRICE :
<img width="1116" height="627" alt="image" src="https://github.com/user-attachments/assets/aac09a2a-129f-465e-9387-4c9a8859d927" />


Partial Autocorrelation
<img width="1111" height="368" alt="image" src="https://github.com/user-attachments/assets/4eeb453a-898c-4eee-8cc0-9d58c00f2f87" />


Autocorrelation
<img width="1122" height="372" alt="image" src="https://github.com/user-attachments/assets/b771edf4-fc73-424e-bd69-44897b60c864" />




SIMULATED ARMA(1,1) PROCESS:
```
ARMA(1,1) Model Summary
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                  Close   No. Observations:                  299
Model:                 ARIMA(1, 0, 1)   Log Likelihood               -1292.650
Date:                Tue, 05 May 2026   AIC                           2593.301
Time:                        14:52:19   BIC                           2608.103
Sample:                    02-01-2000   HQIC                          2599.225
                         - 12-01-2024                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const        270.9818   1454.105      0.186      0.852   -2579.011    3120.975
ar.L1          0.9985      0.007    143.857      0.000       0.985       1.012
ma.L1          0.3439      0.034     10.265      0.000       0.278       0.410
sigma2       325.9373     11.488     28.372      0.000     303.422     348.453
===================================================================================
Ljung-Box (L1) (Q):                   1.67   Jarque-Bera (JB):              1096.05
Prob(Q):                              0.20   Prob(JB):                         0.00
Heteroskedasticity (H):             155.88   Skew:                            -0.36
Prob(H) (two-sided):                  0.00   Kurtosis:                        12.35
===================================================================================
```
<img width="1118" height="602" alt="image" src="https://github.com/user-attachments/assets/5db7fbd3-e617-4d3a-96dc-b11e413eb614" />


Partial Autocorrelation
<img width="1125" height="578" alt="image" src="https://github.com/user-attachments/assets/9ea53433-7a91-4371-835e-f6cf97f7c08b" />


Autocorrelation
<img width="1102" height="575" alt="image" src="https://github.com/user-attachments/assets/3e2fcb63-1082-4768-ae34-b38c8146deb8" />




SIMULATED ARMA(2,2) PROCESS:
```
ARMA(2,2) Model Summary
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                  Close   No. Observations:                  299
Model:                 ARIMA(2, 0, 2)   Log Likelihood               -1285.987
Date:                Tue, 05 May 2026   AIC                           2583.975
Time:                        14:52:19   BIC                           2606.178
Sample:                    02-01-2000   HQIC                          2592.861
                         - 12-01-2024                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const        270.9491   1825.490      0.148      0.882   -3306.946    3848.844
ar.L1          0.6679      0.160      4.179      0.000       0.355       0.981
ar.L2          0.3306      0.158      2.088      0.037       0.020       0.641
ma.L1          0.6366      0.163      3.914      0.000       0.318       0.955
ma.L2         -0.0637      0.089     -0.717      0.473      -0.238       0.110
sigma2       311.5642     10.903     28.576      0.000     290.195     332.934
===================================================================================
Ljung-Box (L1) (Q):                   0.10   Jarque-Bera (JB):              1231.31
Prob(Q):                              0.75   Prob(JB):                         0.00
Heteroskedasticity (H):             146.64   Skew:                            -0.36
Prob(H) (two-sided):                  0.00   Kurtosis:                        12.92
===================================================================================
```
<img width="1117" height="597" alt="image" src="https://github.com/user-attachments/assets/68c105a9-d1e3-4e96-9508-65ddc70ba125" />


Partial Autocorrelation
<img width="1122" height="581" alt="image" src="https://github.com/user-attachments/assets/45f0d6c6-e3a5-45bc-babe-95af6398312e" />



Autocorrelation
<img width="1102" height="582" alt="image" src="https://github.com/user-attachments/assets/e25ff9ac-0a23-436d-afc7-503e4fc36b86" />


### RESULT:
Thus, a python program is created to fir ARMA Model successfully.
