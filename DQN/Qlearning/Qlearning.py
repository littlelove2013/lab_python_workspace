import numpy as np

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
    for i in range(eposide):
        #每次开始随机选择一个起始[0,5]
        state0=np.random.randint(6)
        print('\neposide %d\n'%(i),'path:%d'%(state0),end='')
        step=0
        #随机选择的概率逐步下降
        randrate=baserand+(1-i/eposide)*(1-baserand)
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
            print(' ---> %d'%(state0),end='')
            step+=1
        print('\nrun step:%d, Q value:%d, found rate:%f'%(step,Q.sum(),randrate),'\nQ=\n',Q)


def main():
    print('Qlearning')
    Qlearning()

if __name__ == '__main__':
    main()