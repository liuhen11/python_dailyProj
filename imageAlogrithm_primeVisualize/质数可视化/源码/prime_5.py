import matplotlib.pyplot as plt
from math import sqrt
import math

def is_prime(n):
    x = [p for p in range(2, n+1) if 0 not in [p % d for d in range(2, int(sqrt(p)) + 1)]]
    print(x)
    return x

primesList = is_prime(100000)

pr = []
heshu = []
ziranshu = []
x = []
y = []
x1 = []
y1 = []
x2 = []
y2 = []

for i in range(500,10000):
    if i in primesList:
        x.append(i*math.cos(i))
        y.append(i*math.sin(i))
    else:
        x1.append(i * math.cos(i))
        y1.append(i * math.sin(i))
    x2.append(i * math.cos(i))
    y2.append(i * math.sin(i))

colors1 = '#00CED1'
colors2 = '#DC143C'
plt.figure()
plt.scatter(x, y,c=colors1)
plt.axis('equal')
plt.show()
plt.figure()
plt.scatter(x1, y1,c=colors2)
plt.axis('equal')
plt.show()
plt.figure()
plt.scatter(x1, y1,c=colors2)
plt.scatter(x, y,c=colors1)
plt.axis('equal')
plt.show()