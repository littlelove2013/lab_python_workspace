import numpy as np
import scipy.signal as ss

def conv2withstride(m,filter,stride=(1,1),start=None,gridnum=20):
    tmp=ss.convolve2d(m,filter,'same',boundary='wrap')
    if start==None:
        start=(int(stride[0]/2),int(stride[1]/2))
    r=np.arange(0,gridnum)*stride[0]+start[0]
    c=np.arange(0,gridnum)*stride[0]+start[1]
    grid=np.zeros([gridnum,gridnum])
    for i in range(gridnum):
        grid[i,:]=tmp[r[i],c]
    return grid

def main():
    a=np.ones([6,6])
    b=np.ones([2,2])
    c=conv2withstride(a,b,gridnum=2)
    print(a,b,c)

if __name__ == '__main__':
    main()