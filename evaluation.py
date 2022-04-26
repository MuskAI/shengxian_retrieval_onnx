#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：shengxian_retrieval_onnx 
@File    ：evaluation.py
@IDE     ：PyCharm 
@Author  ：haoran
@Date    ：2022/4/15 10:17 
'''

# from apis.infer import infer
import os,sys

import numpy as np

sys.path.append('./apis')

from apis.construct_database import Database
from model_zoo.resnet50 import ResNetFeat
from model_zoo.mobilenet_v2_md5 import MobileNetV2Feat
from apis.infer import infer
import math
import matplotlib.pyplot as plt
import constants as cst

def inference(db,img, model_path):
    '''

    :param db: saved database
    :param img: test img path
    :return: result
    '''

    method = MobileNetV2Feat(model_path=model_path)
    samples = db.get_samples()
    # print('The length of samples:',len(samples))

    query = method.make_single_sample(img)
    # print('img to be tested:', query)
    # print('sum of hist',sum(query['hist']))
    # parameters
    topd = 10
    topk = 5
    d_type = 'cosine' # distance ty   pe  you can choose 'd1 , d2 , d3  ... d8' and 'cosine' and 'square'
    top_cls, result, std_result = infer(query, samples=samples, depth=topd, d_type=d_type, topk=topk,thr=2.0)
    # print('topk possible predicted classes:', top_cls)
    # print('topd nearest distance matching from the database:', result)
    # print('按阈值过滤后的结果:', std_result)
    return top_cls,std_result

class EvluationFresh:
    def __init__(self):
        self.db = Database(img_dir='database_kh_test', cache_dir='./cache')
        project_root = os.getcwd()
        self.model_path = os.path.join(project_root, 'model_zoo/checkpoint/mobv2-circleloss-imagenet-e10.onnx')
        self.db.connect_db()
        self.img_dir = 'database_kh_test'
        self.reporter = {'Title':'测试模型{},测试数据{}'.format(self.model_path,self.img_dir),
                         'std_out':[]}
        self.read()
        self.write()


    def read(self):
        cls_name_list = os.listdir(self.img_dir)
        std_result_per_cls = []

        for cls_name in cls_name_list:
            count = 3
            if cls_name == '.DS_Store':
                continue
            path = os.path.join(self.img_dir, cls_name)
            img_list = os.listdir(path)
            img_list = [os.path.join(path, name) for name in img_list]

            _out_per_cls = []
            for img_path in img_list:
                if count>0:
                    pass
                else:
                    break
                count = count-1
                _top_cls, _std_result = inference(self.db, img_path, self.model_path)
                _out_per_cls.append({
                    'query_img':img_path,
                    'query_cls':cls_name,
                    'std_out':_std_result
                })
            std_result_per_cls.append({
                'cls_name':cls_name,
                'out_per_cls':_out_per_cls
            })

        self.std_result_per_cls = std_result_per_cls

    def write(self):
        """
        读取返回结果，分析返回结果，并记录
        """
        assert len(self.std_result_per_cls)>0

        self.__draw_angle()


        # for idx,_out_per_cls in enumerate(self.std_result_per_cls):
        #
        #
        #     # 处理好画图需要的参数
        #     self.
        #


    # def arcosine(self,cosine_distance):
    #     """
    #     计算arcosine，得到角度，显示在二维坐标系中
    #     """
    #     angle = []
    #     if isinstance(cosine_distance,float):
    #         cosine_similarity = 1 - cosine_distance
    #         angle.append(math.acos(cosine_similarity))
    #     elif isinstance(cosine_distance,list):
    #         for i in cosine_distance:
    #             cosine_similarity = 1 - i
    #             angle.append(math.acos(cosine_similarity))
    #     else:
    #         pass
    #
    #     return angle

    def __draw_angle(self):
        """
        根据角度画图，按照每一个类别进行画极坐标图
        """

        r=0.8
        for idx,item in enumerate(self.std_result_per_cls):
            for idx2,item2 in enumerate(item['out_per_cls']):
                std_out = item2['std_out']

                LAMBADA = {'丰水梨': '#1f77b4', '杨桃': '#ff7f0e', '芦柑': '#d62728', '苹果': '#bcbd22', '香梨': '#17becf'}
                fig = plt.figure(figsize=(10, 10))
                ax = plt.gca(projection='polar')
                ax.set_thetagrids(np.arange(0.0, 360.0, 15.0))
                ax.set_thetamin(0.0)  # 设置极坐标图开始角度为0°
                ax.set_thetamax(360.0)  # 设置极坐标结束角度为180°
                ax.set_rgrids(np.arange(0, 5000.0, 1000.0))
                ax.set_rlabel_position(0.0)  # 标签显示在0°
                ax.set_rlim(0.0, 1.0)  # 标签范围为[0, 5000)
                # ax.set_yticklabels(['0', '1000', '2000', '3000', '4000', '5000'])

                ax.grid(True, linestyle="-", color="k", linewidth=0.5, alpha=0.5)
                ax.set_axisbelow('True')  # 使散点覆盖在坐标系之上

                print('正确的类别是{},匹配的结果{}'.format(item['cls_name'],std_out))
                for idx3,item3 in enumerate(std_out):
                    c = ax.scatter(math.acos((1-item3['dis']))*5, r, c=LAMBADA[item3['cls']], s=30, cmap='cool', alpha=0.75)
                plt.savefig('极坐标优化后*5/{}-{}.png'.format(item['cls_name'],idx2))


    def __repr__(self):
        return '生鲜模型评估测试'

if __name__ == '__main__':
    evaler = EvluationFresh()