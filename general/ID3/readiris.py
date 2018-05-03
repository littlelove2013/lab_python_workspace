import numpy as np

class PRdata:
    def __init__(self,data_file):
        self.data_file = data_file
        self.readData()
    def readData(self):
        self.Data = []
        self.Label=[]
        # i=0
        dataFile = open(self.data_file)
        data=[]
        for line in open(self.data_file):
	        l = line.strip().split()
	        if len(l) < 6:
		        continue
	        self.Data.append(l[1:-1])
	        self.Label.append(l[-1])
        dataFile.close()
        # self.Data=np.array(self.Data,np.float64)
        # self.Label = np.array(self.Label, np.int32)
def getLabelData():
	filename = 'data'
	data = PRdata(filename)
	#将数据数字化
	dictory=[{'Overcast':0, 'Rain':1, 'Sunny':2}, {'Cool':0, 'Mild':1, 'Hot':2}, {'High':0, 'Normal':1}, {'Weak':0, 'Strong':1}]
	labeldic={'Yes':1,"No":0}
	ldata=[]
	llable=[]
	for i in  range(len(data.Data)):
		da=data.Data[i]
		la=data.Label[i]
		llable.append(labeldic[la])
		tmp=[]
		for i in range(len(da)):
			tmp.append(dictory[i][da[i]])
		ldata.append(tmp)
	return ldata,llable,dictory,labeldic
if __name__ == '__main__':
    ldata,label,dictory,labeldic=getLabelData()
    print(ldata)
    print(label)