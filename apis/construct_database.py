"""
@author:haoran
version: beta 2.0
功能：从图片目录构建数据库构建数据库，并提供增删改查的接口
"""
import pandas as pd
import os
from hashlib import md5
import numpy as np
import time
from six.moves import cPickle
import re
from tqdm import tqdm
from model_zoo.mobilenet_v2_md5 import MobileNetV2Feat


class Database(object):
    """
    1. 创建数据库
    2. 连接数据库
    3. 增删改查接口
    """

    def __init__(self,img_dir='../database',cache_dir='../cache'):
        self.img_dir = img_dir
        self.cache_dir = cache_dir
        self.RES_model = 'mobilenetV2'
        self.pick_layer = 'avgPooling'
        sample_cache = '{}-{}-{}'.format(self.RES_model, self.pick_layer, 'md5')
        self.db_path = os.path.join(self.cache_dir, sample_cache)

        self.samples = None

    def create_db(self):
        # 如果数据库文件存在，则不再重新创建，对于需要增删的需求，需要通过增删改查的接口实现
        # 如果希望重新创建数据库文件，则需要删除原数据库文件

        db_path = self.db_path
        img_dir = self.img_dir
        if os.path.exists(db_path):
            print('数据库已经存在')
            return None
        else:
            # 导入模型
            method = MobileNetV2Feat(model_path='../model_zoo/checkpoint/ret_mobilenet_v2.onnx')
            samples = []
            # 开始遍历图片
            for root, _, files in tqdm(os.walk(img_dir, topdown=False)):
                cls = root.split('/')[-1]
                for name in files:
                    if not name.endswith('.png') and not name.endswith('.jpg'):
                        continue
                    img = os.path.join(root, name)

                    # 开始编码
                    md5_code = self.get_md5(img_path=img)

                    query = method.make_single_sample(img, verbose=False, md5_encoding=False)

                    # 开始构建数据库中的项
                    sample = {
                        'md5': md5_code,
                        'img': img,
                        'cls': cls,
                        'hist': query['hist'],
                        'data_type': 0,  # 0 is system data , 1 is user data
                        'error_times': 0  # 出错次数，默认为0
                    }
                    samples.append(sample)

            # 保存数据库文件
            sample_cache = '{}-{}-{}'.format(self.RES_model, self.pick_layer, 'md5')
            cPickle.dump(samples, open(os.path.join(self.cache_dir, sample_cache), "wb", True))
    def connect_db(self):
        # 如果有数据库文件直接load
        samples = cPickle.load(open(self.db_path, "rb", True))

        for sample in samples:
            sample['hist'] /= np.sum(sample['hist'])  # normalize

        self.samples = samples
    def get_md5(self, img_path):
        """
        对图片进行md5编码，编码成功则返回str，失败则返回None
        Args:
            img_path: [str] 需要编码图片的地址

        """
        assert os.path.isfile(img_path)

        try:
            name = img_path.split('/')[-1]
            if 'md5' in name:
                # 如果编码已经在文件名中，则可以直接从图片名称中获得编码
                md5_code = name.split('-')[-1].split('.')[0]

            else:
                with open(img_path, 'rb') as img:
                    md5_code = md5(img.read()).hexdigest()
                    os.rename(img_path, os.path.join(img_path.replace(name,''),
                                                     '{}-md5-{}.{}'.format(name.split('.')[0],md5_code,name.split('.')[-1])))
        except Exception as e:
            return None

        return md5_code
    def insert(self, img_path_list=[], img_label_list=[]):
        """
        实现增量学习，在数据库中增加新的特征
        有两种方式实现:1.输入对应图片地址和其类别标签 2. 自动检查需要增加的图片  默认使用第二种方式
        Args:
            img_path_list : 图片路径列表 e.g.['./test.jpg','./demo.png']
        """

        assert self.samples, 'connect to db error'
        # 使用默认的方式开始增量学习
        img_info_list = []
        if len(img_path_list) == 0 or len(img_label_list) == 0:
            # 开始遍历图片，数据库
            for root, _, files in tqdm(os.walk(self.img_dir, topdown=False)):
                cls = root.split('/')[-1]
                for name in files:
                    if not name.endswith('.png') and not name.endswith('.jpg'):
                        continue
                    img = os.path.join(root, name)

                    # 开始编码, 修改这里还可以提速
                    md5_code = self.get_md5(img_path=img)

                    img_info_list.append({
                        'md5': md5_code,
                        'img': img,
                        'cls': cls,
                    })

            # 检查重复项
            diffs = self.check_duplicate(samples=self.samples, img_info_list=img_info_list)

            # 生成需要新增的图片列表
            img_path_list = [i['img'] for i in diffs['insert']]
            img_label_list = [i['cls'] for i in diffs['insert']]

        method = MobileNetV2Feat(model_path='../model_zoo/checkpoint/ret_mobilenet_v2.onnx')

        for idx, item in enumerate(img_path_list):
            # 首先判断路径是否存在
            if os.path.isfile(item):
                # 开始增量学习
                query = method.make_single_sample(d_img=item)
                # 增加到数据库
                sample = {
                    'md5': self.get_md5(item),
                    'img': item,
                    'cls': img_label_list[idx],
                    'hist': query['hist'],
                    'data_type': 1,  # 0 is system data , 1 is user data
                    'error_times': 0  # 出错次数，默认为0
                }
                self.samples.append(sample)
        # 如果有修改则重新保存数据库
        if len(img_label_list) !=0 and len(img_label_list) !=0 :
            cPickle.dump(self.samples, open(self.db_path, "wb", True))
            print('\033[41m增量学习成功！数据库已更新！')
    def check_duplicate(self, samples, img_info_list):
        """
        通过文件目录与数据库中对比,找出多余的，和缺失的项

        Args:
             samples: 数据库，是一个list
             data_csv: 与文件目录保持一致

             return  diffs = [{
            'insert': insert_list,
            'del': del_list
        }]
        """
        img_md5_list = [i['md5'] for i in img_info_list]
        img_md5_set = set(img_md5_list)
        samples_md5_list = [i['md5'] for i in samples]
        samples_md5_set = set(samples_md5_list)

        # 求两个集合的交、差集
        c_s = list(img_md5_set.difference(samples_md5_set))  # csv_set - samples_set 新增的数据
        s_c = list(samples_md5_set.difference(img_md5_set))  # samples_set - csv_set 删除的数据

        # 需要添加的图片信息
        insert_list = []
        for i in c_s:
            insert_list.append({
                'md5':img_info_list[img_md5_list.index(i)]['md5'],
                'img':img_info_list[img_md5_list.index(i)]['img'],
                'cls':img_info_list[img_md5_list.index(i)]['cls'],
            })

        # 从samples中找出需要删除的信息
        del_list = []
        for i in s_c:
            del_list.append(samples[samples_md5_list.index(i)])

        diffs = {
            'insert': insert_list,
            'del': del_list
        }
        return diffs
    def delete(self):
        pass
    def update(self):
        pass
    def find(self):
        pass
    def get_samples(self):
        return self.samples
    def __len__(self):
        pass
    def __repr__(self):
        return "DATABASE VERSION IS BETA 1.0 "

if __name__ == '__main__':
    db = Database()

    # 创建数据库，如果数据库文件存在则需要删除才能创建
    db.create_db()


    # 连接数据库，使用之前都需要连接数据库
    # db.connect_db()
    # db.insert()