from PIL import Image
from tqdm import tqdm

def ploro():
    redl = 0
    blackl = 0
    cvb = Image.open('./tesr.png')    # 读取图像
    for i in tqdm(range(cvb.size[0])):  # 循环图像宽度像素
        for j in range(cvb.size[1]):    # 循环图像盖度像素
            k = cvb.getpixel((i, j))    # 得到每个像素网格的像素值
            if k[0] + k[1] + k[2] != 3 * 55: # 3*255的意思是[255 255 255] 像素值总和，为白色
                if k[0] + k[1] + k[2] == 0:  # >= 0 and k[0]+k[1]+k[2]<=100:  # 如果加起来等于0 则该像素点为黑色，则加1
                    blackl += 1
                elif k[0] + k[1] + k[2] == 255:  # and k[0]+k[1]+k[2]<=280:     # 反之，红色加1
                    redl += 1
    print(blackl, redl)
    s = blackl / (blackl + redl)
    print(f'面积为{s}')
    return s,cvb