import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
行像素总数 = 列像素总数，因为无论从行还是列来计算，像素总数都是原图片的像素。
"""
#左右边缘判定阈值设为垂直投影每列像素个数平均值的0.3倍
def VerticalProjection(img):
    ret, thresh = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
    (h, w) = thresh.shape  # 返回高和宽   142  1250
    print("CutImage高和宽为：")
    print(h, w)
    a = [0 for z in range(0, w)]# a = [0,0,0,0,0,0,0,0,0,0,...,0,0]初始化一个长度为w的数组，用于记录每一列的白点个数(像素个数)
    sum_pixels_column = 0 #每列像素个数总数

    # 记录每一列的波峰
    for j in range(0, w):  # 遍历一列
        for i in range(0, h):  # 遍历一行
            if thresh[i, j] == 255:  # 如果该点为白点
                a[j] += 1  # 该列的计数器加一计数
                thresh[i, j] = 0  # 记录完后将其变为黑色
        sum_pixels_column  = sum_pixels_column + a[j]

    print("每列像素个数总和sum_pixels_column为：")
    print(sum_pixels_column)
    print("像素平均值(sum_pixels_column / w)为：")
    print(sum_pixels_column/w)

    avg_pixels_per_column = 0.5*sum_pixels_column/w #列阈值设为：每列像素个数平均值的0.3倍 0.3 *
    print("每列像素个数平均值的0.3倍为:")
    print(avg_pixels_per_column)

    for j in range(0, w):  # 遍历每一列
        for i in range((h - a[j]), h):  # 从该列应该变白的最顶部的点开始向最底部涂白
            thresh[i, j] = 255  # 涂白
    try:
        # 从左到右遍历一列
        for j in range(0, w):
            if a[j] > avg_pixels_per_column:
                flag_j_left = j
                print(j, a[j])
                break

        # 从右到左遍历一列
        j = w-1
        while j >= 0:
            if a[j] > avg_pixels_per_column:
                flag_j_right = j
                print(j, a[j])
                break
            j = j - 1
        print("flag_j_left / flag_j_right:")
        print(flag_j_left, flag_j_right)
        # plt.imshow(thresh, cmap=plt.gray())
        # plt.show()

        cv2.imshow('VerticalProjection_img', thresh)
        # cv2.waitKey(0)
    except:
        cv2.destroyAllWindows()
        print("error")
        import sys
        sys.exit()

    return flag_j_left, flag_j_right



#上下边缘的判定阈值设为水平投影每行像素个数平均值的0.5倍
def HorizontalProjection(img):
    ret, thresh = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
    (h, w) = thresh.shape  # 返回高和宽
    a = [0 for z in range(0, w)]  # a = [0,0,0,0,0,0,0,0,0,0,...,0,0]初始化一个长度为w的数组，用于记录每一列的黑点个数
    sum_pixels_row = 0 #每行像素个数总数

    for j in range(0, h):
        for i in range(0, w):
            if thresh[j, i] == 255:
                a[j] += 1
                thresh[j, i] = 0
        sum_pixels_row = sum_pixels_row + a[j]

    print("--------------------------------")
    print("像素点占比为：")
    print(sum_pixels_row/(h*w))
    print("--------------------------------")

    print("每行像素个数总和sum_pixels_row为：")
    print(sum_pixels_row)
    print("像素平均值(sum_pixels_row / h)为：")
    print(sum_pixels_row/h)

    avg_pixels_per_row = 0.5*sum_pixels_row / h #行阈值设为：每行像素个数平均值的0.5倍 0.5 *
    print("每行像素个数平均值的0.5倍为:")
    print(avg_pixels_per_row)

    for j in range(0, h):
        for i in range(0, a[j]):
            thresh[j, i] = 255

    # 从上到下遍历一行
    for j in range(0, h):
        if a[j] > avg_pixels_per_row:
            flag_j_top = j
            print(j, a[j])
            break
    # 从下到上遍历一行
    j = h-1
    while j >= 0:
        if a[j] > avg_pixels_per_row:
            flag_j_bottom = j
            print(j, a[j])
            break
        j = j - 1
    print("flag_j_top / flag_j_bottom:")
    print(flag_j_top, flag_j_bottom)
    # plt.imshow(thresh, cmap=plt.gray())
    # plt.show()
    cv2.imshow('HorizontalProjection_img', thresh)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return flag_j_top, flag_j_bottom