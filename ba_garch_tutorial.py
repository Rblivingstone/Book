import datetime as dt
import sys

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from arch import arch_model

start = dt.datetime(2000,1,1)
end = dt.datetime(2017,1,1)
sp500 = web.get_data_google('SPY', start=start, end=end)
returns = 100 * sp500['Close'].pct_change().dropna()
returns.plot()
plt.show()

model=arch_model(returns, vol='Garch', p=1, o=0, q=1, dist='Normal')
results=model.fit()
print(results.summary())

forecasts = results.forecast(horizon=30, method='simulation')
sims = forecasts.simulations

lines = plt.plot(sims.values[-1,::30].T, alpha=0.33)
lines[0].set_label('Simulated paths')
plt.plot()

print(np.percentile(sims.values[-1,30].T,5))
plt.hist(sims.values[-1,30],bins=50)
plt.title('Distribution of Returns')