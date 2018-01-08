import Func
import dogbreed
import VGG_beta3 as VGG_train
import VGG_pre
import os

def main():
    print('-----------------------------------')
    print('please see readme.txt first!')
    print('-----------------------------------')
    #判断是否存在保存的模型，如果存在则直接预测，不存在则先訓練模型
    root = dogbreed.root
    savefilepath = root + 'modelsave/'
    if not os.path.exists(savefilepath + '/checkpoint'):  # 判断模型是否存在
        print('error:', savefilepath, " not exist!")
        print('You must train a model before predicting!')
        VGG_train.train()
    #训练完之后，进行预测
    VGG_pre.VGG_pre()

if __name__ == '__main__':
    main()