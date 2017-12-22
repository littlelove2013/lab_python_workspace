import numpy as np
<<<<<<< HEAD
import matplotlib.pyplot as plt
#定义状态类

=======
>>>>>>> b4f1d6be65a76c79ed8ecf08ec35ea4428d9e97e

def Qlearning():
    #reward矩阵
    R=np.array([
        [-1, -1, -1, -1, 0, -1],
        [-1, -1, -1, 0, -1, 100],
        [-1, -1, -1, 0, -1, -1],
        [-1, 0, 0, -1, 0, -1],
        [0, -1, -1, 0, -1, 100],
        [-1, 0, -1, -1, 0, 100],
    ])
    Q=np.array([
        [-1, -1, -1, -1, 0, -1],
        [-1, -1, -1, 0, -1, 0],
        [-1, -1, -1, 0, -1, -1],
        [-1, 0, 0, -1, 0, -1],
        [0, -1, -1, 0, -1, 0],
        [-1, 0, -1, -1, 0, 0],
    ])
    eposide=100#做10次测试
    lamda=0.8
    baserand=0.2#每次做随机跳转还是做取最大Q值的跳转，
<<<<<<< HEAD
    randlist=np.zeros([eposide])
=======
>>>>>>> b4f1d6be65a76c79ed8ecf08ec35ea4428d9e97e
    for i in range(eposide):
        #每次开始随机选择一个起始[0,5]
        state0=np.random.randint(6)
        print('\neposide %d\n'%(i),'path:%d'%(state0),end='')
        step=0
        #随机选择的概率逐步下降
<<<<<<< HEAD
        randrate=baserand+(1-i/eposide)*(1-baserand)
        randlist[i]=randrate
=======
        randrate=baserand+(1-(i+1)/(eposide))*(1-baserand)
>>>>>>> b4f1d6be65a76c79ed8ecf08ec35ea4428d9e97e
        while(state0!=5):
            rate=np.random.rand()
            max_action=0
            if rate<randrate:
                #随机选择下一步
                area=np.where(R[state0]!=-1)[0]
                max_action=area[np.random.randint(len(area))]
            else:
                #选择使当前state0行最大值的action去执行，并得到下一个state
                max_action=np.where(Q[state0] == Q[state0].max())[0][0]
            #根据R矩阵更新Q矩阵
            Q[state0,max_action]=R[state0,max_action]+lamda*max(Q[max_action])
            #state0指向max_action转向的state,此处就是max_action
            state0=max_action
<<<<<<< HEAD
            print(' ---> %d'%(state0),end='')
            step+=1
        print('\nrun step:%d, Q value:%d, found rate:%f'%(step,Q.sum(),randrate),'\nQ=\n',Q)

    plt.plot(np.arange(0,eposide),randlist)
    plt.grid()
    plt.show()
=======
            print(' ---> %d:rate(%.4f)'%(state0,rate),end='')
            step+=1
        print('\nrun step:%d, Q value:%d, found rate:%f'%(step,Q.sum(),randrate))

>>>>>>> b4f1d6be65a76c79ed8ecf08ec35ea4428d9e97e

def main():
    print('Qlearning')
    Qlearning()

if __name__ == '__main__':
    main()