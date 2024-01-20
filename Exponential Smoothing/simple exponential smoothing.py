import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

x1 = np.linspace(0, 1, 100)
y1 = pd.Series(np.multiply(x1, (x1 - 0.5)) + np.random.randn(100))
ets1 = SimpleExpSmoothing(y1)
r1 = ets1.fit()
pred1 = r1.predict(start=len(y1), end=len(y1) + len(y1) // 2)

pd.DataFrame({
    'origin': y1,
    'fitted': r1.fittedvalues,
    'pred': pred1
}).plot(legend=True)
plt.show()