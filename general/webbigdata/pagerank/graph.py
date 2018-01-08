# -- coding: utf-8 --
# 
# PageRank工程的Graph创建程序
#
from collections import deque
class Graph():

	def __init__(self):
		self.node_n={}

	def add_nodes(self,nodelist):
		for i in nodelist:
			self.add_node(i)
	#对每一个节点,建一个链表(数组),链表(数组)保存的是其指向的节点
	def add_node(self,node):
		if not node in self.nodes():
			self.node_n[node]=[]

	def add_edge(self,edge):
		u,v=edge
		if (v not in self.node_n[u]):# and (u not in self.node_n[v]):#为什么要求u not in v呢?
			self.node_n[u].append(v)#u->v
		#if u !=v:
		#	self.node_n[v].append(u)
	#获取dict的关键字集合,即节点name
	def nodes(self):
		return self.node_n.keys()

