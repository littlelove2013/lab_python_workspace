import readiris as rd
import copy
# 最一般假设
def getG():
	G=[[{'Sunny','Overcast','Rain'},{'Hot','Mild','Cool'},{'High','Normal'},{'Weak','Strong'}]]
	return G
# 最特殊假设
def getS():
	S=[set() for i in range(4)]
	return S
#负样本降低G的一般性
def downG(G,S,h):
	newG=[]
	for i in range(len(G)):
		for j in range(len(h)):
			g=copy.deepcopy(G[i])
			if h[j] in g[j]:
				g[j].remove(h[j])
			if more_genneral_than_or_equal_to(g,S):#判断特殊性
				newG.append(g)
	return newG
#正样本增加S的一般性
def upS(G,S,h):
	for i in range(len(h)):
		S[i].add(h[i])#特殊性不断向上
	# 判断特殊性
	newG=[]
	for i in range(len(G)):
		g=copy.deepcopy(G[i])
		if more_genneral_than_or_equal_to(g,S):#判断特殊性
			newG.append(g)
	return newG,S
# G应该比S更一般
def more_genneral_than_or_equal_to(g,S):
	for i in range(len(g)):
		if g[i]<S[i]:
			return False
	return True
# 候选消除法
def CandidateElimilation(G,S,data):
	Data=data.Data
	Label = data.Label
	for i in range(len(Data)):
		if Label[i]=='Yes':#正例
			G,S=upS(G,S,Data[i])
			if len(G)==0:
				print("G=", G)
				return {}
		elif Label[i]=='No':#反例
			G=downG(G,S,Data[i])
			if len(G)==0 :
				print("G=",G)
				return {}
		print("D%d %s:"%(i+1,Label[i]),Data[i],"\nS%d:"%(i+1),S,"\nG%d:"%(i+1),G[0],'......%d more elments.....'%(len(G)))
	return {}
if __name__ == '__main__':
    print("Candidate elimation method!")
    G=getG()
    S=getS()
    print('G:',G,'\nS:',S)

    filename = 'data'
    data = rd.PRdata(filename)
    # print(data.Data)
    # print(data.Label)
    CandidateElimilation(G, S, data)