import matplotlib.pyplot as plt
from math import sqrt
from math import fabs
from matplotlib.font_manager import FontProperties
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def get_first_factor(p):
    return int(sqrt(p))


def get_second_factor(p):
    p = p - int(sqrt(p)) * int(sqrt(p))
    q = sqrt(p)
    return q


def get_third_factor(p):
    p = p - int(sqrt(p)) * int(sqrt(p))
    q = sqrt(p)
    p = p - int(sqrt(q)) * int(sqrt(q))
    q = sqrt(p)
    return q


font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)
plt.title(u'example:prime = a^2 + b^2 + c', FontProperties=font)

lst_f = []
lst_s = []
lst_t = []

for i in [p for p in range(2, 50000) if 0 not in [p % d for d in range(2, int(sqrt(p))+1)]]:
    f = int(get_first_factor(i))
    s = int(get_second_factor(i))
    t = int(get_third_factor(i))
    v = i - f * f - s * s - t * t
    if v > t:
        t += 1
        v = i - f * f - s * s - t * t
    lst_f.append(f)
    lst_s.append(s)
    lst_t.append(t)
    # print(f"{i} = {f}^2 + {s}^2 + {v}")

    # plt.scatter(f, s, s=int(fabs(v)) * 3, c="#ff1212", marker='o')

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(lst_f, lst_s, lst_t)
ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})

plt.show()
