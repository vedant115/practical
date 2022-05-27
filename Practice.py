# ------------ BFS --------------------
'''
graph = {
  'A' : ['B','C'],
  'B' : ['D', 'E'],
  'C' : ['F'],
  'D' : [],
  'E' : ['F'],
  'F' : []
}

# print(graph)

visited = []
queue = []

def bfs(graph, visited, node):
    visited.append(node)
    queue.append(node)
    
    while queue:
        n = queue.pop(0)
        print(n, end=" ")
        for nei in graph[n]:
            if nei not in visited:
                visited.append(nei)
                queue.append(nei)
                
bfs(graph, visited, 'A')
'''
# -------------- DFS -------------
'''
graph = {
  'A' : ['B','C'],
  'B' : ['D', 'E'],
  'C' : ['F'],
  'D' : [],
  'E' : ['F'],
  'F' : []
}

visited = set()

def dfs(visited, node):
    if node not in visited:
        print(node)
        visited.add(node)

        for nei in graph[node]:
            dfs(visited, nei)
    
dfs(visited, 'A')
'''
# ----------- PRIMS --------------
'''
G = [[0, 9, 75, 0, 0],
     [9, 0, 95, 19, 42],
     [75, 95, 0, 51, 66],
     [0, 19, 51, 0, 31],
     [0, 42, 66, 31, 0]]

V = len(G)
no_edge = 0
INF = 9999999
selected = [0]*V

selected[0] = True
print("edge    weight")
while no_edge < (V-1):
  x=0
  y=0
  minimum = INF
  for i in range(V):
    if selected[i]:
      for j in range(V):
        if (not selected[j]) and G[i][j]:
          if minimum > G[i][j]:
            minimum = G[i][j]
            x=i
            y=j

  print(str(x)+"-"+str(y)+"   "+str(G[x][y]))
  selected[y] = True
  no_edge+=1
  '''

  # --------- KRUSKAL --------------
  '''
  def find(graph, node):
	if graph[node] < 0:
		return node
	else:
		temp = find(graph, graph[node])
		graph[node] = temp
		return temp

def union(graph, a, b, ans):
	x=a
	y=b

	a = find(graph, a)
	b = find(graph, b)

	if a == b:
		pass
	else:
		ans.append([x, y])

		if graph[a] < graph[b]:
			graph[a] += graph[b]
			graph[b] = a
		else:
			graph[b] += graph[a]
			graph[a] = graph[b]


inpt = [[1,2,1], [1,3,3], [2,6,4], [3,6,2], [3,4,1], [4,5,5], [6,5,6], [6,7,2], [5,7,7]]
n = 7
ans = []
inpt = sorted(inpt, key=lambda x: x[2])
graph= [-1]*(n+1)
for u, v, w in inpt:
	union(graph, u, v, ans)

for item in ans:
	print(item)


# ----------------- Graph Coloring ---------------
# Python program for solution of M Coloring
# problem using backtracking

class Graph():

	def __init__(self, vertices):
		self.V = vertices
		self.graph = [[0 for column in range(vertices)] for row in range(vertices)]

	# A utility function to check
	# if the current color assignment
	# is safe for vertex v
	def isSafe(self, v, colour, c):
		for i in range(self.V):
			if self.graph[v][i] == 1 and colour[i] == c:
				return False
		return True
	
	# A recursive utility function to solve m
	# coloring problem
	def graphColourUtil(self, m, colour, v):
		if v == self.V:
			return True

		for c in range(1, m + 1):
			if self.isSafe(v, colour, c) == True:
				colour[v] = c
				if self.graphColourUtil(m, colour, v + 1) == True:
					return True

	def graphColouring(self, m):
		colour = [0] * self.V
		if self.graphColourUtil(m, colour, 0) == None:
			return False

		# Print the solution
		print ("Solution exist and Following are the assigned colours:")
		for c in colour:
			print (c,end=' ')
		return True

# Driver Code
g = Graph(4)
g.graph = [[0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 1], [1, 0, 1, 0]]
m = 3
g.graphColouring(m)
'''

# ----------- A * ---------------
'''
# https://stackabuse.com/basic-ai-concepts-a-search-algorithm/
from collections import deque

class Graph:
    # example of adjacency list (or rather map)
    # adjacency_list = {
    # 'A': [('B', 1), ('C', 3), ('D', 7)],
    # 'B': [('D', 5)],
    # 'C': [('D', 12)]
    # }

    def __init__(self, adjacency_list):
        self.adjacency_list = adjacency_list

    def get_neighbors(self, v):
        return self.adjacency_list[v]

    # heuristic function with equal values for all nodes
    def h(self, n):
        H = {
            'A': 1,
            'B': 1,
            'C': 1,
            'D': 1
        }

        return H[n]

    def a_star_algorithm(self, start_node, stop_node):
        # open_list is a list of nodes which have been visited, but who's neighbors
        # haven't all been inspected, starts off with the start node
        # closed_list is a list of nodes which have been visited
        # and who's neighbors have been inspected
        open_list = set([start_node])
        closed_list = set([])

        # g contains current distances from start_node to all other nodes
        # the default value (if it's not found in the map) is +infinity
        g = {}

        g[start_node] = 0

        # parents contains an adjacency map of all nodes
        parents = {}
        parents[start_node] = start_node

        while len(open_list) > 0:
            n = None

            # find a node with the lowest value of f() - evaluation function
            for v in open_list:
                if n == None or g[v] + self.h(v) < g[n] + self.h(n):
                    n = v

            if n == None:
                print('Path does not exist!')
                return None

            # if the current node is the stop_node
            # then we begin reconstructin the path from it to the start_node
            if n == stop_node:
                reconst_path = []

                while parents[n] != n:
                    reconst_path.append(n)
                    n = parents[n]

                reconst_path.append(start_node)

                reconst_path.reverse()

                print('Path found: {}'.format(reconst_path))
                return reconst_path

            # for all neighbors of the current node do
            for (m, weight) in self.get_neighbors(n):
                # if the current node isn't in both open_list and closed_list
                # add it to open_list and note n as it's parent
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight

                # otherwise, check if it's quicker to first visit n, then m
                # and if it is, update parent data and g data
                # and if the node was in the closed_list, move it to open_list
                else:
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        parents[m] = n

                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add(m)

            # remove n from the open_list, and add it to closed_list
            # because all of his neighbors were inspected
            open_list.remove(n)
            closed_list.add(n)

        print('Path does not exist!')
        return None


adjacency_list = {
    'A': [('B', 1), ('C', 3), ('D', 7)],
    'B': [('D', 5)],
    'C': [('D', 12)]
}
graph1 = Graph(adjacency_list)
graph1.a_star_algorithm('A', 'D')


'''

'''
def greet(bot_name, birth_year):
    print("Hello! My name is {0}.".format(bot_name))
    print("I was created in {0}.".format(birth_year))


def remind_name():
    print('Please, remind me your name.')
    name = input()
    print("What a great name you have, {0}!".format(name))


def guess_age():
    print('Let me guess your age.')
    print('Enter remainders of dividing your age by 3, 5 and 7.')

    rem3 = int(input())
    rem5 = int(input())
    rem7 = int(input())
    age = (rem3 * 70 + rem5 * 21 + rem7 * 15) % 105

    print("Your age is {0}; that's a good time to start programming!".format(age))


def count():
    print('Now I will prove to you that I can count to any number you want.')
    num = int(input())

    counter = 0
    while counter <= num:
        print("{0} !".format(counter))
        counter += 1


def test():
    print("Let's test your programming knowledge.")
    print("Why do we use methods?")
    print("1. To repeat a statement multiple times.")
    print("2. To decompose a program into several small subroutines.")
    print("3. To determine the execution time of a program.")
    print("4. To interrupt the execution of a program.")

    answer = 2
    guess = int(input())
    while guess != answer:
        print("Please, try again.")
        guess = int(input())

    print('Completed, have a nice day!')
    print('.................................')
    print('.................................')
    print('.................................')


def end():
    print('Congratulations, have a nice day!')
    print('.................................')
    print('.................................')
    print('.................................')
    input()
    
greet('SBot', '2022')  # change it as you need
remind_name()
guess_age()
count()
test()
end()
'''
