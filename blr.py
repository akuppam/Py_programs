import numpy as np
print 'resid sum of sq: %.2f' % np.mean((model.predict(X)-y) ** 2)
