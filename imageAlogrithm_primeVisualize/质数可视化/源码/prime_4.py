import numpy as np
from PIL import Image
from math import sqrt
from math import fabs


def find_prime_list(n):
    """返回不大于n的素数组成的numpy数组"""

    nums = np.arange(n + 1)  # 生成0到n的数组
    nums[1] = 0  # 数组第1和元素置0，从2开始，都是非0的
    k, m = 0, pow(n + 1, 0.5)

    while True:  # 循环
        primes = nums[nums > 0]  # 找出所有非0的元素
        if primes.any():  # 如果存在不为0的元素
            p = primes[k]  # 则第一个元素为素数
            k += 1
            nums[2 * p::p] = 0  # 这个素数的所有倍数，都置为0
            if p > m:  # 如果找到的素数大于上限的平方根
                break  # 则退出循环
        else:
            break  # 全部0，也退出循环

    return nums[nums > 0]


def get_square(n):
    """将从1开始的n个连续的整数排成一个列表"""

    side = int(pow(n, 1 / 2))  # 方阵边长

    if side % 2:
        row, col = side - 1, side - 1
        direct = 'left'
    else:
        row, col = 0, 0
        direct = 'right'

    result = [[None for j in range(side)] for i in range(side)]

    for i in range(n, 0, -1):
        result[row][col] = i

        if direct == 'right':
            if col + 1 == side or result[row][col + 1]:  # 如果不能继续向右，则向下:
                row += 1
                direct = 'down'
            else:  # 否则继续向右
                col += 1
        elif direct == 'down':
            if row + 1 == side or result[row + 1][col]:  # 如果不能继续向下，则向左
                col -= 1
                direct = 'left'
            else:  # 否则继续向下
                row += 1
        elif direct == 'left':
            if col - 1 < 0 or result[row][col - 1]:  # 如果不能继续向左，则向上
                row -= 1
                direct = 'up'
            else:  # 否则继续向左
                col -= 1
        elif direct == 'up':
            if row - 1 < 0 or result[row - 1][col]:  # 如果不能继续向上，则向右
                col += 1
                direct = 'right'
            else:  # 否则继续向上
                row -= 1

    return np.array(result)



def plot_prime(side):
    """绘制不大于side*side的质数分布图"""

    n = side * side
    square = get_square(n)
    primes = find_prime_list(n)
    #primes = [p for p in range(2, side+1) if 0 not in [p % d for d in range(2, int(sqrt(p)) + 1)]]
    im_arr = np.isin(square, primes).astype(np.uint8) * 255
    im = Image.fromarray(im_arr)
    im.save('%dx%d.jpg' % (side, side))

plot_prime(1000)