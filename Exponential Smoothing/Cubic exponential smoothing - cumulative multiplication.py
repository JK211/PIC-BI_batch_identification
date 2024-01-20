import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing

x3 = np.linspace(0, 4 * np.pi, 100)
y3 = pd.Series(20 + 0.1 * np.multiply(x3, x3) + 8 * np.cos(2 * x3) + 2 * np.random.randn(100))
ets3 = ExponentialSmoothing(y3, trend="multiplicative", seasonal='add', seasonal_periods=25)
r3 = ets3.fit()
pred3 = r3.predict(start=len(y3), end=len(y3) + len(y3) // 2)

pd.DataFrame({
    'origin': y3,
    'fitted': r3.fittedvalues,
    'pred': pred3
}).plot(legend=True)
plt.show()