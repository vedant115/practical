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
