#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：shengxian_retrieval_onnx 
@File    ：mnn_demo.py
@IDE     ：PyCharm 
@Author  ：haoran
@Date    ：2022/4/23 15:50 
'''
import MNN.expr as F
from torchvision import transforms
from PIL import Image

mnn_model_path = './pipeline-small-秤.mnn'
image_path = './test.jpg'

vars = F.load_as_dict(mnn_model_path)
inputVar = vars["input"]
# 查看输入信息
print('input shape: ', inputVar.shape)
# print(inputVar.data_format)

# 写入数据
input_image = Image.open(image_path)
preprocess = transforms.Compose([ transforms.Resize((320,320)), transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]), transforms.ToTensor()])

if (inputVar.data_format == F.NC4HW4):
        inputVar.reorder(F.NCHW)
input_tensor = preprocess(input_image)
inputVar.write(input_tensor.tolist())
with open('test.txt','a') as file0:
    print(vars, file=file0)
# 查看输出结果
outputVar = vars['output']
print('output shape: ', outputVar.shape)
# print(outputVar.read())

cls_id = F.argmax(outputVar, axis=1).read()
cls_probs = F.softmax(outputVar, axis=1).read()

print("cls id: ", cls_id)
print("cls prob: ", cls_probs[0, cls_id])