# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 20:17:22 2019

@author: GEORGE
"""
import os
import numpy as np
from PIL import Image

count = np.zeros(10)
images = []
def func(x,y,sigma=1):
    return 100*(1/(2*np.pi*sigma))*np.exp(-((x-2)**2+(y-2)**2)/(2.0*sigma**2))

def splitimage(src, rownum, colnum):
    img = Image.open(src)
    
    w, h = img.size
    if rownum <= h and colnum <= w:
        print('Original image info: %sx%s, %s, %s' % (w, h, img.format, img.mode))
        print('开始处理图片切割, 请稍候...')
        s = os.path.split(src)
        fn = s[1].split('.')
        basename = fn[0]
        num = 0
        rowheight = h // rownum
        colwidth = w // colnum
        i = 0
        basename = basename[:len(basename)-1]
        for r in range(rownum):
            for c in range(colnum):
                box = (c * colwidth, r * rowheight, (c + 1) * colwidth, (r + 1) * rowheight)
                sd = basename[i]
                if sd != '_':
                    img.crop(box).save(os.path.join("./cutImage/",str(num) +'.png'),"PNG")
                    num = num + 1
                i += 1
        print('图片切割完毕，共生成 %s 张小图片。' % num)
    else:
        print('不合法的行列切割参数！')



