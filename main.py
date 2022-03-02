
import tkinter.messagebox
from tkinter import filedialog as fd
from tkinter import *
import networkx as nx
import matplotlib.pyplot as plt
import sys


# fig, ax = plt.subplots(figsize=(6,4))
#
#
# def update(num):
    # ax.clear()
    # i = num // 3
    # j = num % 3 + 1
    # # triad = sequence_of_letters[i:i+3]
    # # path = ["O"] + ["".join(sorted(set(triad[:k + 1]))) for k in range(j)]
    #
    # # Background nodes
    # nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color="gray")
    # null_nodes = nx.draw_networkx_nodes(G, pos=pos, nodelist=set(G.nodes()) - set(path), node_color="white",  ax=ax)
    # null_nodes.set_edgecolor("black")
    #
    # # Query nodes
    # query_nodes = nx.draw_networkx_nodes(G, pos=pos, nodelist=path, node_color=idx_colors[:len(path)], ax=ax)
    # query_nodes.set_edgecolor("white")
    # nx.draw_networkx_labels(G, pos=pos, labels=dict(zip(path,path)),  font_color="white", ax=ax)
    # edgelist = [path[k:k+2] for k in range(len(path) - 1)]
    # nx.draw_networkx_edges(G, pos=pos, edgelist=edgelist, width=idx_weights[:len(path)], ax=ax)
    #
    # # Scale plot ax
    # ax.set_title("Frame %d:    "%(num+1) +  " - ".join(path), fontweight="bold")
    # ax.set_xticks([])
    # ax.set_yticks([])

#
# ani = plt.animation.FuncAnimation(fig, update, frames=6, interval=1000, repeat=True)
# plt.show()















def filing(Input):
    filename = Input
    with open(filename) as f:
        lines = f.readlines()
        lines = (line for line in lines if line)

    count = 0
    list1 = []
    Node = []

    for line in lines:
        count += 1
        if not line.strip():
            continue
        else:
            listli = line.split()
            list1.append(listli)

    v = int(list1[1][0])

    adjacent = [[0] * v for _ in range(v)]


    for i in range(0, v):
        ps = (float(list1[2 + i][1]), float(list1[2 + i][2]))
        Node.append(ps)
        l=0

    for i in range(v + 2, len(list1) - 1):
        noe = int(list1[i][0])
        for j in range(1,len(list1[i]),4):
            t = int(list1[i][j])
            w = float(list1[i][j + 2])
            adjacent[l][t]=w
        l+=1
    source = int(list1[len(list1)-1][0])
    return source,adjacent,v,Node

def printResultGraph(fileName,graph):
    graph = returnUndirectedGraph(graph)
    s,adjMat,v,pos = filing(fileName)

    g = nx.Graph()

    for i in range(0, v):
        po = (pos[i][0], pos[i][1])
        g.add_node(i, pos=po)

    for i in range(0, v):
        for j in range(0, v):
            if (adjMat[i][j] != 0):
                weight = adjMat[i][j] / 10000000
                if graph[i][j]!=0:
                    g.add_edge(i, j, weight=weight,color='r')
                else:
                    g.add_edge(i, j, color='white')
    edges = g.edges()
    i=0
    j=0
    colors = [g[i][j]['color'] for i, j in edges]
    label = nx.get_edge_attributes(g, 'weight')
    pos = nx.get_node_attributes(g, 'pos')
    # all_edges = set(
    #     tuple(sorted((n1, n2))) for n1, n2 in g.edges()
    # )
    nx.draw(g,edge_color=colors, pos=pos, with_labels=1, node_size=200, width=1)
    nx.draw_networkx_edge_labels(g, pos, edge_labels=label, font_size=10, font_family="sans-serif")
    plt.show()


def returnUndirectedGraph(graph):
    for i in range(0,len(graph)):
        for j in range(0,len(graph[i])):
          if(graph[i][j]!=0 and graph[i][j]!=graph[j][i]):
              min = graph[i][j]
              if(graph[j][i]!=0 and graph[j][i]<min):
                  min = graph[j][i]
              graph[i][j]=min
              graph[j][i]=min
    return graph;


def printUnDirectedGraph(adjMat,pos,v):
    g = nx.Graph()

    for i in range(0,v):
        po = (pos[i][0],pos[i][1])
        g.add_node(i,pos=po)

    for i in range(0,v):
        for j in range(0,v):
            if(adjMat[i][j]!=0):
                weight = adjMat[i][j]/10000000
                g.add_edge(i,j,weight=weight)

    label = nx.get_edge_attributes(g, 'weight')

    pos = nx.get_node_attributes(g, 'pos')

    nx.draw(g, pos, with_labels=1, node_size=200, width=1, edge_color="b")
    nx.draw_networkx_edge_labels(g, pos, edge_labels=label, font_size=10, font_family="sans-serif")
    plt.show()







def printInputGraph():


    nodes_data = []
    edges_data = []
    nodes=0
    start=0
    temp_1=[]
    temp_2=[]
    f_read = open(filename1, "r")
    str = f_read.read()
    f_read.close()
    str = str.replace("NETSIM", "")
    line_sep = str.split("\n")
    no_line= [line.strip() for line in line_sep if line.strip() != ""]
    str_no_line = ""
    for line in no_line:
        str_no_line += line + "\n"
    str = str_no_line
    temp = str.splitlines()
    temp = [i.split("\t") for i in temp]
    nodes = int(temp[0][0])
    start = int(temp[-1][0])
    temp.pop(0)
    temp.pop()
    for j in range(nodes):
        temp_1 = [float(i) for i in temp[0]]
        nodes_data.append(tuple(temp_1[0:3]))
        temp.pop(0)
    temp_1.clear()
    for i in range(len(temp)):
        for j in range(2, len(temp[i]), 2):
            temp_1.append(temp[i][j])
        for j in range(len(temp_1)):
            for k in range(1, len(temp[i])):
                if temp[i][k] == temp_1[j]:
                    del temp[i][k]
                    break
        temp_1.clear()
    l=0
    for i in temp:
        length=int(i.pop(0))
        for j in range(length):
            edges_data.append((int(nodes_data[l][0]),int(i[0]),float(i[1])))

            i.pop(0)
            i.pop(0)
        l=l+1
    for i in range(len(edges_data)):
        if edges_data[i][0]==edges_data[i][1]:
            continue
        temp_2.append((edges_data[i][0],edges_data[i][1],int(edges_data[i][2])))
    edges_data.clear()
    for i in temp_2:
        edges_data.append(i)
    temp_2.clear()

    g=nx.Graph()

    for i in range(nodes):
        g.add_node(int(nodes_data[i][0]), pos=(nodes_data[i][1], nodes_data[i][2]))

    for i in range(len(edges_data)):
        g.add_edge(int(edges_data[i][0]), int(edges_data[i][1]), weight=(int((edges_data[i][2]) / 10000000)))
    label = nx.get_edge_attributes(g, 'weight')

    pos = nx.get_node_attributes(g, 'pos')

    nx.draw(g, pos, with_labels=1, node_size=200, width=1, edge_color="b")
    nx.draw_networkx_edge_labels(g, pos, edge_labels=label, font_size=10, font_family="sans-serif")
    plt.show()














def findMaxVertex(visited, weights,V):
    index = -1;
    maxW = sys.maxsize;
    for i in range(V):
        if (visited[i] == False and weights[i] < maxW):
            maxW = weights[i];
            index = i;
    return index;



def PrimsAlgo(graph,V,S):
    visited = [True] * V;
    weights = [0] * V;
    parent = [0] * V;

    for i in range(V):
        visited[i] = False;
        weights[i] = sys.maxsize;

    weights[S] = 0;
    parent[S] = -1;

    for i in range(V - 1):
        maxVertexIndex = findMaxVertex(visited, weights,V);
        visited[maxVertexIndex] = True;
        for j in range(V):
            if (graph[j][maxVertexIndex] != 0 and visited[j] == False):
                if (graph[j][maxVertexIndex] < weights[j]):
                    weights[j] = graph[j][maxVertexIndex];
                    parent[j] = maxVertexIndex;
    mst=0
    for i in range(V):
            for j in range(V):
                graph[i][j]=0
                if(parent[j]==i):
                    graph[i][j]=weights[j]
                    mst=mst+weights[j]
    mst=mst/10000000
    tkinter.messagebox.showinfo("MST Cost",mst)
    return graph




def kunion(i, j,parent):
    a = find(parent,i)
    b = find(parent,j)
    parent[a] = b


def kruskalMST(cost,V):
    G = [[0] * V for _ in range(V)]
    parent=[0]*V

    for i in range(V):
        parent[i] = i

    edge_count = 0
    mst=0
    while edge_count < V - 1:
        min = sys.maxsize
        a = -1
        b = -1
        for i in range(V):
            for j in range(V):
                if find(parent,i) != find(parent,j) and cost[i][j] < min and cost[i][j]!=0:
                    min = cost[i][j]
                    a = i
                    b = j
        mst=mst+min
        kunion(a, b,parent)
        G[a][b]=min
        G[b][a] = min
        edge_count += 1
    mst=mst/10000000
    tkinter.messagebox.showinfo("MST cost",mst)
    return G




def BellmenFord(graph, V, src):
    dist = [sys.maxsize] * V
    dist[src] = 0
    parent = [-1]*V
    totaldist = 0
    for q in range(V - 1):
        for i in range(V):
            for j in range(V):
                if graph[i][j] == 0:
                    continue
                x = i
                y = j
                w = graph[i][j]

                if dist[x] + w < dist[y]:
                    dist[y] = dist[x] + w
                    parent[y]=x

    for i in range(V):
        for j in range(V):
            if graph[i][j] == 0:
                continue
            x = i
            y = j
            w = graph[i][j]
            if dist[x] != sys.maxsize and dist[x] + w < dist[y]:
                return None

    for i in range(V):
        for j in range(V):
            graph[i][j]=0

    for i in range(V):
        if(parent[i]!=-1):
            graph[parent[i]][i] = dist[i]
            totaldist=totaldist+dist[i]
    totaldist=totaldist/10000000
    tkinter.messagebox.showinfo("Total Distance",totaldist)
    return graph








def minDistance(V, dist, sptSet):
    min = sys.maxsize
    min_index = -1

    # Search not nearest vertex not in the
    # shortest path tree
    for u in range(V):
        if dist[u] < min and sptSet[u] == False:
            min = dist[u]
            min_index = u

    return min_index


def dijkstra(G, V, src):
    dist = [sys.maxsize] * V
    sptSet = [False] * V
    parent = [-1] * V
    dist[src] = 0
    totaldist=0.00
    for cout in range(V):

        x = minDistance(V, dist, sptSet)
        sptSet[x] = True


        for y in range(0, V):

            if (G[x][y] > 0 and sptSet[y] == False) and (dist[y] > dist[x] + G[x][y]):
                dist[y] = dist[x] + G[x][y]
                parent[y] = x
    for i in range(V):
        for j in range(V):
            G[i][j] = 0
    for i in range(V):
        if (parent[i] != -1):
            G[parent[i]][i] = dist[i]
            totaldist= totaldist+dist[i]
    tkinter.messagebox.showinfo("total distance",totaldist)
    return G


def showResultFunc():
    s, adjM, v, p = filing(filename1)
    if Selecteddrop.get() == algorithms[0]:

        adjM = PrimsAlgo(adjM,v,s)
        printResultGraph(filename1,adjM)
    elif Selecteddrop.get() == algorithms[3]:

        adjM = BellmenFord(adjM, v, s)
        printResultGraph(filename1,adjM)
    elif Selecteddrop.get()==algorithms[2]:

        adjM = dijkstra(adjM,v,s)
        printResultGraph(filename1,adjM)
    elif Selecteddrop.get()==algorithms[4]:

        adjM = floydWarshal(adjM)
        printResultGraph(filename1,adjM)
    elif Selecteddrop.get()==algorithms[5]:

        Clustering_Coefficient(adjM,p,v)
    elif Selecteddrop.get()==algorithms[6]:

        adjM = boruvka(adjM,v)
        printResultGraph(filename1,adjM)
    elif Selecteddrop.get()==algorithms[1]:

        adjM = kruskalMST(adjM,v)
        printResultGraph(filename1,adjM)



def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])

def union(parent, rank, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)

    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1


def boruvka(graph, V):
    parent = []
    rank = []
    G  = [[0] * V for _ in range(V)]

    cheapest = []

    numTrees = V
    MSTweight = 0

    for node in range(V):
        parent.append(node)
        rank.append(0)
        cheapest = [-1] * V


    while numTrees > 1:

        for i in range(V):
            for j in range(V):
                if graph[i][j] == 0:
                    continue
                w = graph[i][j]
                set1 = find(parent, i)
                set2 = find(parent, j)

                if set1 != set2:

                    if cheapest[set1] == -1 or cheapest[set1][2] > w:
                        cheapest[set1] = [i, j, w]

                    if cheapest[set2] == -1 or cheapest[set2][2] > w:
                        cheapest[set2] = [i, j, w]
        for node in range(V):


            if cheapest[node] != -1:
                u, v, w = cheapest[node]
                set1 = find(parent, u)
                set2 = find(parent, v)

                if set1 != set2:
                    MSTweight += w
                    union(parent, rank, set1, set2)
                    G[u][v] = w
                    # print("Edge %d-%d with weight %d included in MST" % (u, v, w))
                    numTrees = numTrees - 1


        cheapest = [-1] * V
    MSTweight=MSTweight/10000000
    tkinter.messagebox.showinfo("Total Mst cost",MSTweight)
    return G







def floydWarshal(graph):
    dist = graph
    for i in range(len(graph)):
        for j in range(len(graph)):
            if(i==j):
                dist[i][j] = 0
            elif(graph[i][j]==0):
                dist[i][j] = sys.maxsize
            else:
                dist[i][j]=graph[i][j]
    for k in range(len(graph)):
        for i in range(len(graph)):
            for j in range(len(graph)):
                dist[i][j] = min(dist[i][j],dist[i][k] + dist[k][j])
    for i in range(len(graph)):
        for j in range(len(graph)):
            if(dist[i][j]==sys.maxsize):
                graph[i][j] = 0
            else:
                graph[i][j] = dist[i][j]
    return graph



def Clustering_Coefficient(adjM,p,v):
    g = nx.Graph()

    for i in range(0, v):
        po = (p[i][0], p[i][1])
        g.add_node(i, pos=po)

    for i in range(0, v):
        for j in range(0, v):
            if (adjM[i][j] != 0):
                weight = adjM[i][j]
                g.add_edge(i, j, weight=weight)
    p = nx.average_clustering(g)
    tkinter.messagebox.showinfo("Clustering Coefficient",p)


#Graphical Interface

# filing("input10.txt")
backGroundColor = '#1e3153'
textColor = 'white'

root = Tk()
root.geometry("700x700")
root.configure(bg=backGroundColor)

IntroLabel = Label(root, text="Please select alogorithm and input file from drop downs", fg=textColor, bg='white',
                   font=('Courier', 13))
IntroLabel.pack(fill='x')

horizontal_layout_dropDowns = Frame(root)
algorithms = [
    "Prims",
    "Kruskal",
    "Dijkstra",
    "Bellmen Ford",
    "Floyd Warshal",
    "Clustering Coefficient",
    "Boruvka's algorithm"
]
Selecteddrop = StringVar()
Selecteddrop.set(algorithms[0])
AlogDropDown = OptionMenu(horizontal_layout_dropDowns, Selecteddrop, *algorithms)
AlogDropDown.config(bg='grey', fg='white', font=('Courier', 13))
AlogDropDown.grid(row=5, column=1, pady=20, padx=10)
filename1 = StringVar()
def select_file():
    filetypes = (
        ('text files', '*.txt'),
    )
    global filename1
    filename1 = fd.askopenfilename(
        title='Open a file',
        initialdir='/',
        filetypes=filetypes)

open_button = Button(
    root,
    text='Select File',
    command=select_file
)

open_button.pack(expand=True)
horizontal_layout_dropDowns.configure(bg=backGroundColor, pady=100)
horizontal_layout_dropDowns.pack()

horizontal_layout_button = Frame(root)
ShowInput = Button(horizontal_layout_button, text="Show graph", bg='white', fg='black', padx=5, pady=5,
                   font=('Courier', 13), command=printInputGraph)
ShowInput.grid(row=10, column=2, padx=10, pady=20)

ShowResult = Button(horizontal_layout_button, text="Calculate", bg='white', fg='black', padx=5, pady=5,
                    font=('Courier', 13),command=showResultFunc)
ShowResult.grid(row=10, column=5, padx=10, pady=20)
horizontal_layout_button.configure(bg=backGroundColor)
horizontal_layout_button.pack()
root.mainloop()



