import numpy as np
import matplotlib.pyplot as plt
def data_generation(n1,n2,n3):
    x1=np.random.multivariate_normal((1,1),[[2,0],[0,2]],n1)
    x2=np.random.multivariate_normal((4,4),[[2,0],[0,2]],n2)
    x3=np.random.multivariate_normal((8,1),[[2,0],[0,2]],n3)
    re=np.r_[x1,x2,x3]
    return re
X1=data_generation(333,333,334)
X2=data_generation(600,300,100)

plt.scatter([x[0] for x in X2],[x[1] for x in X2])
plt.show()

