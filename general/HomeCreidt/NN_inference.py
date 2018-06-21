import tensorflow as tf
import numpy as np
from funcs import *
import NN
import os
import pandas as pd

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def inference(Test,Test_id,filename='Test'):
	with tf.Session() as sess:
		# 判断模型是否存在
		if not os.path.exists('save/model.model.meta'):
			print("先进行训练")
		saver = tf.train.import_meta_graph('save/model.ckpt-100.meta')
		saver.restore(sess, tf.train.latest_checkpoint('./save'))
		y = tf.get_collection('pred_network')[0]
		#accuracy=tf.get_collection('accuracy')[0]
		graph = tf.get_default_graph()
		input_x = graph.get_operation_by_name('input_x').outputs[0]
		pred=graph.get_operation_by_name('pred_network').outputs[0]
		keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
		# 使用y进行预测
		feed={input_x: Test, keep_prob: 1.0}
		prediction=sess.run(pred, feed_dict=feed)
	#保存到csv
	savedata=pd.DataFrame([Test_id,prediction[:,1]],columns=['SK_ID_CURR','TARGET'])
	savedata.to_csv(filename, index=False)
	print('save %s to file!' % (filename))
	
if __name__ == '__main__':
	filemat = getMat(filename='application', prefix='../input/')
	Test = filemat['Te']
	Test_id = filemat['Te_id']
	Test_id=Test_id.reshape(-1)
	inference(Test,Test_id,'NN_pred')