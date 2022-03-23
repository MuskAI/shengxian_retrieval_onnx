"""
划分数据集用于准确度测试
"""
from tqdm import tqdm
import os,shutil
import random
def split_data(img_dir,percent_cls=0.5,save_dir='./database_train'):
    # 开始遍历图片

    for root, _, files in tqdm(os.walk(img_dir, topdown=False)):
        cls = root.split('/')[-1]

        # split
        for idx, name in enumerate(files):
            img = os.path.join(root, name)
            # 前半部分作为training dataset
            if idx<len(files)*percent_cls:
                if not os.path.exists('./database_train/{}'.format(cls)):
                    os.mkdir('./database_train/{}'.format(cls))
                shutil.copy(img,os.path.join('./database_train/{}'.format(cls),name))
            else:
                if not os.path.exists('./database_test/{}'.format(cls)):
                    os.mkdir('./database_test/{}'.format(cls))
                shutil.copy(img, os.path.join('./database_test/{}'.format(cls), name))







if __name__ == '__main__':
    split_data(img_dir='database(orig)')