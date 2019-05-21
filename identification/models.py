from django.db import models

# Create your models here.
class Picture(models.Model):
    #上传图片的模型类
    pic = models.ImageField(upload_to='uploadimage/')
    def __str__(self):
        return self.pic
