import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import Holt

x2 = np.linspace(0, 99, 100)
y2 = pd.Series(0.1 * x2 + 2 * np.random.randn(100))
ets2 = Holt(y2)
r2 = ets2.fit()
pred2 = r2.predict(start=len(y2), end=len(y2) + len(y2) // 2)

pd.DataFrame({
    'origin': y2,
    'fitted': r2.fittedvalues,
    'pred': pred2
}).plot(legend=True)
plt.show()