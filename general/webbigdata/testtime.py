import numpy as np
import time
r=100
c=600000
start = time.time()
a=np.random.rand(r,c)
b=np.random.rand(r,c)
c=a.dot(b.T)
print('cost time is %.4fs'%(time.time()-start))