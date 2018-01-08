# -- coding: utf-8 --
# 
# PageRank工程的主程序
# 1. 数据导入
# 2. 数据处理和算法运算
# 3. 结果保存
#
import Graph
import pagerank

class PRdata:
    def __init__(self,data_file):
        self.data_file = data_file
        self.readData()
    def readData(self):
        self.List = []
        dataFile = open(self.data_file)
        for line in open(self.data_file):
            A=line.strip().split('\t')[0]
            B=line.strip().split('\t')[1]
            self.List.append(A)
            self.List.append(B)
        dataFile.close()

if __name__ == '__main__':

    PRdata = PRdata("data/WikiData.txt")
    dg = Graph.Graph()
    dg.add_nodes(PRdata.List)
    with open("data/WikiData.txt","r") as f:
         for line in f:
            A=line.strip().split('\t')[0]
            B=line.strip().split('\t')[1]
            dg.add_edge((A, B))
    f.close()
    #print(dg.node_n)
    pr = pagerank.pagerank(dg)
    page_ranks = pr.page_rank()

    print("The final page rank is saving...")
    with open("results/result.txt","w") as f_w:
        for key, value in page_ranks.items():
            #print(key + " : ", value[0])
            f_w.write(key + '\t\t' + str(value[0]) + '\n')
    f_w.close()
    print("The final page rank is okey.")
input('请按任意键退出...')