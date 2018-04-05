import numpy as np
import random
from sets import Set
import matplotlib.pyplot as plt


def random_color():
	r = lambda: random.randint(0,255)
	s = lambda: random.randint(0,100)
	x = random.randint(1, 3)
	if x == 1:
		return ('#%02X%02X%02X' % (s(),r(),r()))
	elif x == 2:
		return ('#%02X%02X%02X' % (r(),s(),r()))
	else:
		return ('#%02X%02X%02X' % (r(),r(),s()))


colors =  'rgbcmykw'

cluster_colors = {}
x = []
y = []
c = []
adj = {}
aggregates = {}

def distance(pointa, pointb):
	return abs(pointa[0] - pointb[0]) + abs(pointa[1] - pointb[1])

def connected(a, b):
	points_a = aggregates[a]
	points_b = aggregates[b]
	for pointa in points_a:
		for pointb in points_b:
			if distance(pointa, pointb) == 1:
				return 1
	return 0

n = int(raw_input("Print number of lines/points in the file"))

for i in range(0, n):
	[xp, yp, clus] = map(lambda x : int(x), raw_input().split())

	if clus not in aggregates:
		aggregates[clus] = [[xp, yp]]
	else:
		aggregates[clus].append([xp, yp])
	adj[clus] = []

for key1 in aggregates:
	cluster_colors[key1] = -1
	for key2 in aggregates:
		if connected(key1, key2):
			adj[key1].append(key2)

import sys
sys.setrecursionlimit(1000000)

counter = 0

def dfs(cluster):
	global counter
	counter = counter + 1
	s = Set()
	for ne in adj[cluster]:
		s.add(cluster_colors[ne])

	toAdd = 1
	while True:
		if toAdd in s:
			toAdd = toAdd + 1
			continue
		else:
			cluster_colors[cluster] = toAdd
			break

	for ne in adj[cluster]:
		if cluster_colors[ne] == -1:
			dfs(ne)

for key in aggregates:
	if cluster_colors[key] == -1:
		dfs(key)

for i in range(1, 101):
	for j in range(1, 101):
		x.append(i)
		y.append(j)
		c.append('r')

for key in aggregates:
	for point in aggregates[key]:
		x.append(point[0])
		y.append(point[1])
		c.append(colors[cluster_colors[key]])

plt.scatter(x, y, s = 4, c = c)
plt.show()
