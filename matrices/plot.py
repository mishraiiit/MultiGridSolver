import matplotlib.pyplot as plt
X = []
Y = []
for i in range(18620):
	inp = map(int, raw_input().split())
	X.append(inp[0])
	Y.append(inp[1])

plt.plot(X, Y, 'ro')
plt.axis([0, 101, 0, 101])
plt.show()
