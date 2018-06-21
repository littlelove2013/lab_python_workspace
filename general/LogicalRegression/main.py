import func

if __name__ == '__main__':
	Train, Trainl, Test, Testl = func.loaddata(filename="spamData.mat")
	func.preLR(Train, Trainl, Test, Testl, prefun=func.Stnd)
	func.preLR(Train, Trainl, Test, Testl, prefun=func.Log)
	func.preLR(Train, Trainl, Test, Testl, prefun=func.Binary)