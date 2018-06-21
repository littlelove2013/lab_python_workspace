# numpy and pandas for data manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as scio

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder

# File system manangement
import os

# Function to calculate missing values by column# Funct
def missing_values_table(df):
	# Total missing values
	mis_val = df.isnull().sum()

	# Percentage of missing values
	mis_val_percent = 100 * df.isnull().sum() / len(df)

	# Make a table with the results
	mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

	# Rename the columns
	mis_val_table_ren_columns = mis_val_table.rename(
		columns={0: 'Missing Values', 1: '% of Total Values'})

	# Sort the table by percentage of missing descending
	mis_val_table_ren_columns = mis_val_table_ren_columns[
		mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
		'% of Total Values', ascending=False).round(1)

	# Print some summary information
	print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
															  "There are " + str(mis_val_table_ren_columns.shape[0]) +
		  " columns that have missing values.")

	# Return the dataframe with missing information
	return mis_val_table_ren_columns
def changeDtype(src,dstdtype='int64'):
    a=src.values
    a=np.array(a,dtype=dstdtype)
    return a
def fillNanData(df,missing_values,show=True):
	#missing_values = missing_values_table(df)
	miss_most_index = missing_values.loc[missing_values['% of Total Values']>=30].index
	#缺失率高于30%的设置为两类
	#app_train[app_train[miss_most_index]].fillna(-1)
	df[miss_most_index].dtypes.value_counts()
	#print(app_train[miss_most_index].select_dtypes(include=['object']).apply(pd.Series.nunique, axis = 0))
	for col in miss_most_index:
		df[col][df[col].notnull()]=1
		df[col][df[col].isnull()]=0
		df[col]=changeDtype(df[col],'int64')
	# 低于30的数据，离散值用多数填充，连续值用均值填充
	miss_less_index = missing_values.loc[missing_values['% of Total Values'] < 30].index
	df[miss_less_index].select_dtypes(include=['float'])
	for col in miss_less_index:
		print(col,df[col].dtype)
		if df[col].dtype == 'object':
			#imputer = Imputer(strategy='most_frequent')
			#imputer.fit(df[col])
			#b = imputer.transform(df[col]).reshape(-1, )
			#print(col, ":", df[col].shape, " trans:", b.shape, df[col].index)
			df[col][df[col].isnull()] = "missing"
		else:
			if show:
				print(df[col].dtype,'fill with %f'%(df[col].mean()))
			df[col][df[col].isnull()] = df[col].mean()
	missing_values_table(df)
	return df

#解决缺失值的问题
def solveNull(app_train,app_test, show=True):
	print("============解决缺失值的问题============")
	missing_values = missing_values_table(app_train)
	app_train = fillNanData(app_train, missing_values, show)
	app_test = fillNanData(app_test, missing_values, show)
	app_test = fillNanData(app_test, missing_values_table(app_test), show)
	return app_train,app_test
#类别分析和对齐
def solveClass(app_train,app_test,show=True):
	# Number of each type of column
	print("============类别分析和对齐============")
	if (show):
		print(app_train.dtypes.value_counts())
		print(app_test.dtypes.value_counts())
		# Number of unique classes in each object column
		print(app_train.select_dtypes(include=['object']).apply(pd.Series.nunique, axis=0))
		print(app_test.select_dtypes(include=['object']).apply(pd.Series.nunique, axis=0))
	# Create a label encoder object
	le = LabelEncoder()
	le_count = 0
	# Iterate through the columns
	for col in app_train:
		if app_train[col].dtype == 'object':
			if (show):
				print(col, ":", len(list(app_train[col].unique())))
			# If 2 or fewer unique categories
			if len(list(app_train[col].unique())) <= 2:
				# Train on the training data
				le.fit(app_train[col])
				# Transform both training and testing data
				app_train[col] = le.transform(app_train[col])
				app_test[col] = le.transform(app_test[col])

				# Keep track of how many columns were label encoded
				le_count += 1
	if (show):
		print('%d columns were label encoded.' % le_count)
	# one-hot encoding of categorical variables
	app_train = pd.get_dummies(app_train)
	app_test = pd.get_dummies(app_test)
	if (show):
		print('Training Features shape: ', app_train.shape)
		print('Testing Features shape: ', app_test.shape)
	# 特征对齐
	train_labels = app_train['TARGET']
	# Align the training and testing data, keep only columns present in both dataframes
	app_train, app_test = app_train.align(app_test, join='inner', axis=1)
	if (show):
		print('Training Features shape: ', app_train.shape)
		print('Testing Features shape: ', app_test.shape)
		print('Testing Features shape: ', app_test.shape)
	app_train['TARGET'] = train_labels
	return app_train, app_test
#相关性分析
def solveCorrelations(app_train,app_test,show=True):
	print("============相关性分析============")
	correlations = app_train.corr()['TARGET']
	abscorrelations = abs(correlations)
	if (show):
		abscorrelations.plot()
		plt.ylabel('correlations')
		plt.show()
	abscorrelations = abscorrelations.sort_values()
	if (show):
		print('Most Positive Correlations: \n', abscorrelations.tail(15))
	return app_train, app_test
#数据归一化
def solveScale(app_train,app_test,show=True):
	# 数据填充与归一化
	print("============数据归一化============")
	from sklearn.preprocessing import MinMaxScaler, Imputer
	# Drop the target from the training data
	if 'TARGET' in app_train:
		train = app_train.drop(['TARGET'], axis=1).copy()
	else:
		train = app_train.copy()
	features = list(train.columns)

	# Copy of the testing data
	test = app_test.copy()
	#去除id列
	if 'SK_ID_CURR' in train:
		train.drop(['SK_ID_CURR'], axis=1)
	if 'SK_ID_CURR' in test:
		test.drop(['SK_ID_CURR'], axis=1)

	# Median imputation of missing values
	imputer = Imputer(strategy='median')

	# Scale each feature to 0-1
	scaler = MinMaxScaler(feature_range=(0, 1))

	# Fit on the training data
	imputer.fit(train)

	# Transform both training and testing data
	train = imputer.transform(train)
	test = imputer.transform(app_test)

	# Repeat with the scaler
	scaler.fit(train)
	train = scaler.transform(train)
	test = scaler.transform(test)
	if(show):
		print('Training data shape: ', train.shape)
		print('Testing data shape: ', test.shape)
	return train, test
#保存 处理好的数据到mat
def saveMat(filename='application', prefix='../input/'):
	csv_filename_train = prefix + filename + '_train.csv'
	csv_filename_test = prefix + filename + '_test.csv'
	print('does not exists saved mat file! \npreprocess from src file(%s and %s)' % (
	csv_filename_train, csv_filename_test))
	if not (os.path.exists(csv_filename_train) or os.path.exists(csv_filename_test)):
		print('cannot find %s & %s!' % (csv_filename_train, csv_filename_test))
		exit(0)
	app_train = pd.read_csv('../input/application_train.csv')
	print('Training data shape: ', app_train.shape)
	# app_train.head()
	# Testing data features
	app_test = pd.read_csv('../input/application_test.csv')
	print('Testing data shape: ', app_test.shape)
	# app_test.head()
	train_labels = app_train['TARGET']
	test_id = app_test['SK_ID_CURR']
	# 解决缺失值的问题
	# print('解决缺失值的问题')
	app_train, app_test = solveNull(app_train, app_test, False)
	# 类别分析和对齐
	# print('类别分析和对齐')
	app_train, app_test = solveClass(app_train, app_test, False)
	# 相关性分析
	# print('相关性分析')
	app_train, app_test = solveCorrelations(app_train, app_test, False)
	# 数据归一化
	# print('数据归一化')
	train, test = solveScale(app_train, app_test, False)
	# save to mat
	mat_filename = prefix + filename + '.mat'
	mat_file = {'Tr': train, 'Tr_l': np.array(train_labels, np.int), 'Te': test, 'Te_id': np.array(test_id)}
	print('保存数据到%s' % (mat_filename))
	scio.savemat(mat_filename, mat_file)
	return mat_file
#读取保存好的mat，如果不存在则调用saveMat
def getMat(filename='application', prefix='../input/'):
	mat_filename = prefix + filename + '.mat'
	if os.path.exists(mat_filename):
		print('exists saved mat file(%s)!' % (mat_filename))
		matfile = scio.loadmat(mat_filename)
		return matfile
	else:
		return saveMat(filename, prefix)

#逻辑回归预测
def LRpre(test,train,train_labels):
	# LR
	from sklearn.linear_model import LogisticRegression

	# Make the model with the specified regularization parameter
	log_reg = LogisticRegression(penalty='l2', C=0.0001, class_weight='balanced', max_iter=500, solver='sag', verbose=1,
								 n_jobs=-1)

	# Train on the training data

	log_reg.fit(train, train_labels)

	# predict 返回两列，第一列为0的概率，第二列为1的概率
	log_reg_pred = log_reg.predict_proba(test)[:, 1]
	return log_reg_pred
def main():
	# List files available
	print(os.listdir("../input/"))

	# Training data
	app_train = pd.read_csv('../input/application_train.csv')
	print('Training data shape: ', app_train.shape)
	app_train.head()
	# Testing data features
	app_test = pd.read_csv('../input/application_test.csv')
	print('Testing data shape: ', app_test.shape)
	app_test.head()
	# 样本不均衡
	target_counts = app_train['TARGET'].value_counts()
	target_ratio0 = target_counts[0] / (target_counts[0] + target_counts[1])
	target_ratio1 = target_counts[1] / (target_counts[0] + target_counts[1])
	print('target_ratio0=', target_ratio0, ', target_ratio1=', target_ratio1)
	app_train['TARGET'].astype(int).plot.hist()
	plt.xlabel('TARGET')
	#缺失值处理
	missing_values = missing_values_table(app_train)
	app_train = fillNanData(app_train, missing_values)
	app_test = fillNanData(app_test, missing_values)
	app_test = fillNanData(app_test, missing_values_table(app_test))
	# 类别分析
	# Number of each type of column
	print(app_train.dtypes.value_counts())
	print(app_test.dtypes.value_counts())
	# Number of unique classes in each object column
	print(app_train.select_dtypes(include=['object']).apply(pd.Series.nunique, axis=0))
	print(app_test.select_dtypes(include=['object']).apply(pd.Series.nunique, axis=0))
	# Create a label encoder object
	le = LabelEncoder()
	le_count = 0

	# Iterate through the columns
	for col in app_train:
		if app_train[col].dtype == 'object':
			print(col, ":", len(list(app_train[col].unique())))
			# If 2 or fewer unique categories
			if len(list(app_train[col].unique())) <= 2:
				# Train on the training data
				le.fit(app_train[col])
				# Transform both training and testing data
				app_train[col] = le.transform(app_train[col])
				app_test[col] = le.transform(app_test[col])

				# Keep track of how many columns were label encoded
				le_count += 1

	print('%d columns were label encoded.' % le_count)
	# one-hot encoding of categorical variables
	app_train = pd.get_dummies(app_train)
	app_test = pd.get_dummies(app_test)
	print('Training Features shape: ', app_train.shape)
	print('Testing Features shape: ', app_test.shape)
	# 特征对齐
	train_labels = app_train['TARGET']
	# Align the training and testing data, keep only columns present in both dataframes
	app_train, app_test = app_train.align(app_test, join='inner', axis=1)
	print('Training Features shape: ', app_train.shape)
	print('Testing Features shape: ', app_test.shape)
	print('Testing Features shape: ', app_test.shape)
	app_train['TARGET'] = train_labels

	#相关性分析
	# Find correlations with the target and sort
	correlations = app_train.corr()['TARGET']
	# Display correlations
	abscorrelations = abs(correlations)
	abscorrelations.plot()
	plt.ylabel('correlations')
	plt.show()
	abscorrelations = abscorrelations.sort_values()
	print('Most Positive Correlations: \n', correlations.tail(15))

	# 数据填充与归一化
	from sklearn.preprocessing import MinMaxScaler, Imputer
	# Drop the target from the training data
	if 'TARGET' in app_train:
		train = app_train.drop(['TARGET'], axis=1).copy()
	else:
		train = app_train.copy()
	features = list(train.columns)

	# Copy of the testing data
	test = app_test.copy()

	# Median imputation of missing values
	imputer = Imputer(strategy='median')

	# Scale each feature to 0-1
	scaler = MinMaxScaler(feature_range=(0, 1))

	# Fit on the training data
	imputer.fit(train)

	# Transform both training and testing data
	train = imputer.transform(train)
	test = imputer.transform(app_test)

	# Repeat with the scaler
	scaler.fit(train)
	train = scaler.transform(train)
	test = scaler.transform(test)

	print('Training data shape: ', train.shape)
	print('Testing data shape: ', test.shape)

	# LR
	from sklearn.linear_model import LogisticRegression

	# Make the model with the specified regularization parameter
	log_reg = LogisticRegression(penalty='l2', C=0.0001, class_weight='balanced', max_iter=500, solver='sag', verbose=1,
								 n_jobs=-1)

	# Train on the training data

	log_reg.fit(train, train_labels)

	# predict 返回两列，第一列为0的概率，第二列为1的概率
	log_reg_pred = log_reg.predict_proba(test)[:, 1]
	submit = app_test[['SK_ID_CURR']]
	submit['TARGET'] = log_reg_pred

	# Save the submission to a csv file
	submit.to_csv('log_reg_baseline.csv', index=False)
	print('save log_reg_baseline.csv to file!')


if __name__ == '__main__':
    #main()
	#读取文件
	print(os.listdir("../input/"))
	# Training data
	app_train = pd.read_csv('../input/application_train.csv')
	print('Training data shape: ', app_train.shape)
	#app_train.head()
	# Testing data features
	app_test = pd.read_csv('../input/application_test.csv')
	print('Testing data shape: ', app_test.shape)
	#app_test.head()
	train_labels = app_train['TARGET']
	# 解决缺失值的问题
	app_train, app_test=solveNull(app_train, app_test)
	# 类别分析和对齐
	app_train, app_test = solveClass(app_train, app_test)
	# 相关性分析
	app_train, app_test = solveCorrelations(app_train, app_test)
	# 数据归一化
	train, test = solveScale(app_train, app_test)
	#逻辑回归预测
	log_reg_pred=LRpre(test,train,train_labels)
	submit = app_test[['SK_ID_CURR']]
	submit['TARGET'] = log_reg_pred

	filename = 'log_reg_baseline1.csv'
	# Save the submission to a csv file
	submit.to_csv(filename, index=False)
	print('save %s to file!' % (filename))