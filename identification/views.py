from django.shortcuts import render
from django.http import HttpResponse
import os
from . import models
# from subprocess import call
from cut import cutting

# Create your views here.

def home(request):
    # 主页面视图
    if request.method == 'POST':
        img = Picture(pic=request.FILES.get('picture'))
        img.save()
        print(img)
        #--------------------------------#
        NumList = cutting.NumRes(img.pic.url)#可能要修改url绝对路径
        # call(["python cutting.py -i" + img.pic.url + '"'])
        # --------------------------------#

    return render(request, 'home.html', locals())

# def img_url(request):
#     if request.method == 'POST':
#         img = Picture(pic=request.FILES.get('picture'))
#
#     return img