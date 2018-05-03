import readiris
import knn

if __name__ == '__main__':
	filename = 'iris.txt'
	data = readiris.PRdata(filename)
	Data=data.Data
	Label=data.Label
	knn.err_rate(Data,Label,rate=0.1)
	knn.crossvalidation(Data, Label,k_cross=10)