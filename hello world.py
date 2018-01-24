import time
import numpy as np

r=100;
c=600000;
s=time.time()
a=np.random.rand(r,c);
b=np.random.rand(c,r);
c=a.dot(b);
t=time.time()-s
print("time:%fs"%(t))