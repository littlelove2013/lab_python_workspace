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
    
if __name__ == '__main__':
    filename='data'
    data=PRdata(filename)
    print(data.Data)
    print(data.Label)