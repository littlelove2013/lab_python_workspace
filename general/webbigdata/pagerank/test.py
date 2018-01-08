import Graph
dg = Graph.Graph()
dg.add_nodes(["A", "B", "C", "D", "E"])
dg.add_edge(("A", "B"))
dg.add_edge(("A", "C"))#比如这两个就无法共存,因为判断C not in A or A not in C会不成环
dg.add_edge(("A", "D"))
dg.add_edge(("B", "D"))
dg.add_edge(("C", "A"))#比如这两个就无法共存,
dg.add_edge(("C", "E"))
dg.add_edge(("D", "E"))
dg.add_edge(("B", "E"))
dg.add_edge(("E", "A"))
print(dg.node_n)