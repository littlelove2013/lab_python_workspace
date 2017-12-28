import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize,imshow
# from imagenet_classes import class_names
import dogbreed
import os
import savelabeltocvs as sl

#载入VGG模型并预测test
def VGG_pre():
    #首先取出每一个test的batch
    lens=dogbreed.testlens
    batchsize=dogbreed.batchsize
    root = dogbreed.root
    # 判断模型保存路径是否存在，不存在就创建
    savefilepath = root + 'modelsave/'
    if not os.path.exists(savefilepath + '/checkpoint'):  # 判断模型是否存在
        print('error:',savefilepath," not exist!")
        return None
    print('prediction!...')
    score=np.array([])
    with tf.Session() as sess:
        print('重构模型')
        saver = tf.train.import_meta_graph(savefilepath+'dogbreed.model.meta')
        print('载入权重参数')
        saver.restore(sess, tf.train.latest_checkpoint(savefilepath))
        print('获取输入输出')
        y = tf.get_collection('pred_network')[0]
        # accuracy = tf.get_collection('accuracy')[0]
        graph = tf.get_default_graph()
        input_x = graph.get_operation_by_name('input_x').outputs[0]
        keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
        #没用到batch_size
        # batch_size = graph.get_operation_by_name('batch_size').outputs[0]

        # input_y = graph.get_operation_by_name('input_y').outputs[0]
        # keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
        showtime=50#50次就显示一次
        for i in range(lens):
            batch=dogbreed.gettest(i,batchsize=batchsize)
            images=batch['images']
            # 使用y进行预测
            feed = {input_x: images,keep_prob:1.0}
            y_conv=sess.run(y,feed_dict=feed)
            #将所有结果连在一起
            if score.size!=0:
                score=np.concatenate((score,y_conv))
            else:
                score=y_conv
            if (i+1)%showtime==0:
                print('doing %d batch prediction!'%(i+1))
    print('prediction have done!')
    Idlist, dogbreedlist = sl.gettestname()
    print('保存到csv')
    sl.save2vsc(Idlist,dogbreedlist,score,savepath='./dataset/',savename='pretestimage')
    print('保存成功')

def main():
    VGG_pre()

if __name__ == '__main__':
    main()