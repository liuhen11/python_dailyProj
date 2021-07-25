
import matplotlib.pyplot as plt
from math import sqrt
from math import fabs
from matplotlib.font_manager import FontProperties


def get_first_factor(p):
    return int(sqrt(p))


def get_second_factor(p):
    p = p - int(sqrt(p)) * int(sqrt(p))
    q = sqrt(p)
    return q


#font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)
plt.title(u'example:prime = a^2 + b^2 + c')

for i in [p for p in range(2, 100000) if 0 not in [p % d for d in range(2, int(sqrt(p))+1)]]:
    f = int(get_first_factor(i))
    s = int(get_second_factor(i))
    v = i - f * f - s * s
    if v > s:
        s += 1
        v = i - f * f - s * s
    # print(f"{i} = {f}^2 + {s}^2 + {v}")

    plt.scatter(f, s, s=int(fabs(v)) * 3, c="#ff1212", marker='o')

plt.show()
