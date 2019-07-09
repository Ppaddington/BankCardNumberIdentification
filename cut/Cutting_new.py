"""
python cutting.py -i "E://6.jpeg"
"""

import cv2
import numpy as np
import argparse
import sys
import os
from PIL import Image
from code import result

o_path = os.getcwd() # 返回当前工作目录
sys.path.append(o_path) # 添加自己指定的搜索路径
sys.path.append("..")
# sys.path.append('D:/learn/python/pycharm/PycharmProjects/CardIdentification/code/result.py')
from cut.transform import four_point_transform




# 利用确定一条直线的两点坐标（x0,y0),(x1,y1)和确定另一条直线的两点坐标(x2,y2),(x3,y3)求两直线的交点坐标
def point(x0, y0, x1, y1, x2, y2, x3, y3):
    a = y1 - y0
    b = x1 * y0 - x0 * y1
    c = x1 - x0
    d = y3 - y2
    e = x3 * y2 - x2 * y3
    f = x3 - x2
    y = float(a * e - b * d) / (a * f - c * d)
    x = float(y * c - b) / a
    pt = (int(x), int(y))
    return pt

# # 参数解析
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
#                 help="path to input image")
# args = vars(ap.parse_args())

# 读取图像
# image = cv2.imread(args["image"])
def NumRes(url):
    # print(sys.path)
    image = cv2.imread(url)

    """
    图像滤波，即在尽量保留图像细节特征的条件下对目标图像的噪声进行抑制，
    是图像预处理中不可缺少的操作，其处理效果的好坏将直接影响到后续图像处理和分析的有效性和可靠性。
    图像滤波的目的
    1，消除图像中混入的噪声 
    2，为图像识别抽取出图像特征
    
    图像滤波的要求
    1，不能损坏图像轮廓及边缘 
    2，图像视觉效果应当更好
    
    滤波器的定义
    滤波器，顾名思义，是对波进行过滤的器件。（摘自网络） 
    以上的定义是针对物理器件的，但对于图像滤波而言显然也是适用的。
    大家都用过放大镜，这里就以此举一个例子：你将放大镜移动的过程中可以看到放大的物体，滤波器就是一个承载着加权系数的镜片，
    这里就是透过镜片可以看到经过平滑处理过的图像，透过镜片以及伴随着镜片的移动你可以逐渐所有的图像部分。
    
    滤波器的种类
    3种线性滤波：方框滤波、均值滤波、高斯滤波 
    2种非线性滤波：中值滤波、双边滤波
    """
    # 高斯滤波
    image1 = cv2.GaussianBlur(image, (7, 7), 0)
    # cv2.imshow("image1", image1)
    # cv2.waitKey(0)
    """
    参数解释： 
    . InputArray src: 输入图像，图像为1、3、4通道的图像，当模板尺寸为3或5时，图像深度只能为CV_8U、CV_16U、CV_32F中的一个，
    如而对于较大孔径尺寸的图片，图像深度只能是CV_8U。 
    . OutputArray dst: 输出图像，尺寸和类型与输入图像一致，可以使用Mat::Clone以原图像为模板来初始化输出图像dst 
    . int ksize: 滤波模板的尺寸大小，必须是大于1的奇数，如3、5、7……
    中值滤波将图像的每个像素用邻域 (以当前像素为中心的正方形区域)像素的 中值 代替 。
    为什么图像会倾斜？？？
    """
    # image1 = cv2.medianBlur(image, ksize=5)   #中值滤波


    # 边缘检测
    edges = cv2.Canny(image1, 50, 150, apertureSize=3)
    # cv2.imshow("first canny", edges)
    # cv2.waitKey(0)

    # 膨胀处理
    closed = cv2.dilate(edges, None, iterations=2)
    # cv2.imshow("first dilate", closed)
    # cv2.waitKey(0)

    #-----------------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------------------------#
    # 霍夫直线检测
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    lines1 = lines[:, 0, :]
    outline = [[100000, 100000, 100000, 100000], [0, 0, 0, 0], [100000, 100000, 100000, 100000], [0, 0, 0, 0]]
    coords = []

    # 直线过滤
    for rho, theta in lines1[:]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        # 取所有确定直线的两点中纵坐标之和最小者为银行卡的确定上边界直线的点
        # 纵坐标之和最大者为银行卡的确定下边界直线的点
        # 横坐标之和最小者为银行卡的确定左边界直线的点
        # 横坐标之和最大者为银行卡的确定右边界直线的点
        if (y1 > 0 and y2 > 0 and y1 + y2 < outline[0][1] + outline[0][3]):
            outline[0] = [x1, y1, x2, y2]
        if (y1 > 0 and y2 > 0 and y1 + y2 > outline[1][1] + outline[1][3]):
            outline[1] = [x1, y1, x2, y2]
        if (x1 > 0 and x2 > 0 and x1 + x2 < outline[2][0] + outline[2][2]):
            outline[2] = [x1, y1, x2, y2]
        if (x1 > 0 and x2 > 0 and x1 + x2 > outline[3][0] + outline[3][2]):
            outline[3] = [x1, y1, x2, y2]

    # 传入上一步循环获得的确定四条边直线的点，获得两两相交的交点
    coords.append((point(outline[0][0], outline[0][1], outline[0][2], outline[0][3], outline[2][0], outline[2][1],
                         outline[2][2], outline[2][3])))
    coords.append((point(outline[0][0], outline[0][1], outline[0][2], outline[0][3], outline[3][0], outline[3][1],
                         outline[3][2], outline[3][3])))
    coords.append((point(outline[1][0], outline[1][1], outline[1][2], outline[1][3], outline[3][0], outline[3][1],
                         outline[3][2], outline[3][3])))
    coords.append((point(outline[1][0], outline[1][1], outline[1][2], outline[1][3], outline[2][0], outline[2][1],
                         outline[2][2], outline[2][3])))

    # 传入交点，进行图像矫正
    warped = four_point_transform(image, coords)
    # 显示矫正结果并保存
    cv2.imshow("Warped", warped)
    # cv2.imwrite("e://warped_img.jpg", warped)
    cv2.waitKey(0)
    print("__________Warped____________")
    #-----------------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------------------------#


    """
    双线性插值法作为尺寸归一化方法。
    void resize(InputArray src, OutputArray dst, Size dsize, double fx=0, double fy=0, int interpolation=INTER_LINEAR );
    INTER_LINEAR	双线性插值（默认设置）
    """
    # 改变图片大小并保存
    resize_wraped = cv2.resize(warped, (700,500))
    cv2.imshow("Resize_Warped_img", resize_wraped)
    # cv2.imwrite("e://re_warped_img.jpg", resize_wraped)
    cv2.waitKey(0)
    print("__________Resize_Warped____________")
    # import sys
    # sys.exit()

    # 切割银行卡号大体位置并保存
    cut_img = resize_wraped[200:330, 50:700]
    cv2.imshow("Cut_img", cut_img)
    # cv2.imwrite("e://cut_img.jpg", cut_img)
    # cv2.waitKey(0)
    print("__________Cut_Img____________")


    # RBG转YUV
    image_yuv = cv2.cvtColor(cut_img,cv2.COLOR_BGR2YUV)
    # cv2.imshow('YUV',image_yuv)
    # cv2.waitKey(0)
    # 得到Y U V三个分量的图像，可知U V分量的图像就是背景干扰图案
    y, u, v = cv2.split(image_yuv)
    # cv2.imshow('y', y)
    # cv2.imshow('u', u)
    # cv2.imshow('v', v)
    # cv2.waitKey(0)

    # 对U V分量的图像进行边缘检测，得到背景干扰图案的像素位置，然后在原图案中减去景干扰图案的像素
    # 对U V分量的边缘图像进行逐点或运算，得到的图像当做背景图案的边缘图像
    #在原图像减去背景图案边缘图像之前先进行一次膨胀处理，得到较粗的边缘图像以增强消除效果
    canny_u = cv2.Canny(u, 5, 45, apertureSize=3)
    # cv2.imshow("canny_u", canny_u)
    canny_v = cv2.Canny(v, 5, 45, apertureSize=3)
    # cv2.imshow("canny_v", canny_v)
    # cv2.waitKey(0)
    #逐点或运算
    img_u_or_v = cv2.bitwise_or(canny_u, canny_v)
    cv2.imshow("img_u_or_v", img_u_or_v)
    cv2.waitKey(0)

    # 对背景图案边缘图像进行一次膨胀处理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    #膨胀图像
    dilated_img_u_or_v = cv2.dilate(img_u_or_v,kernel)
    cv2.imshow("Dilated img_u_or_v Image",dilated_img_u_or_v)#效果很好
    cv2.waitKey(0)

    #获取银行卡图像的边缘图像   apertureSize 就是 Sobel 算子的大小
    #轮廓是白，背景是黑。如何转换为轮廓是黑，背景是白？
    canny_cut_img = cv2.Canny(cut_img, 60, 200, apertureSize=3)
    cv2.imshow("canny_cut_img", canny_cut_img)
    # cv2.waitKey(0)
    print("canny_cut_img")

    # 原边缘图像减去背景图案边缘图像
    canny_cut_img_subtract = cv2.subtract(canny_cut_img,dilated_img_u_or_v)
    cv2.imshow("canny_cut_img_subtract",canny_cut_img_subtract)#效果很好
    cv2.waitKey(0)


    """边缘检测得到银行卡的边缘图像后，为获取卡号边缘的精确范围，
    将边缘图像在水平方向和垂直方向分别进行投影。
    字符所在区域像素点分布较多，据此设定阈值
    """
    # avg_pixels_per_column = VerticalProjection(canny_cut_img)
    # avg_pixels_per_row = HorizontalProjection(canny_cut_img)
    import sys
    try:
        flag_j_left, flag_j_right = VerticalProjection(canny_cut_img_subtract)
        flag_j_top, flag_j_bottom = HorizontalProjection(canny_cut_img_subtract)

        # 将cut_img按照得到的上下左右阈值进行切割
        cut_img_threshold = cut_img[flag_j_top:flag_j_bottom, flag_j_left:flag_j_right]
        cv2.imshow("Cut_img_threshold", cut_img_threshold)
        cv2.imwrite("e://cut_img_threshold.jpg", cut_img_threshold)
        cv2.waitKey(0)
        print("__________Cut_Img__threshold__________")
    except:
        cv2.destroyAllWindows()
        sys.exit()