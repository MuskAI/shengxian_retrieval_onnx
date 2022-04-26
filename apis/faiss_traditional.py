#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：shengxian_retrieval_onnx 
@File    ：faiss_demo.py
@IDE     ：PyCharm 
@Author  ：haoran
@Date    ：2022/4/25 17:19 
'''
import numpy as np
import time
import faiss



from scipy import spatial

if __name__ == '__main__':
    d = 1280  # dimension
    nb = 20000  # database size
    nq = 1  # nb of queries
    np.random.seed(1234)  # make reproducible
    xb = np.random.random((nb, d)).astype('float32')
    # xb_len = np.linalg.norm(xb, axis=1, keepdims=True)
    # xb = xb / xb_len
    xq = np.random.random((nq, d)).astype('float32')
    # xq_len = np.linalg.norm(xq, axis=1, keepdims=True)
    # xq = xq / xq_len
    t1 = time.time()
    nlist = 10  # we want to see 4 nearest neighbors
    dis_list = []
    top5_list = []
    for i in range(10):
        for i in range(len(xb)):
            _dis = spatial.distance.cosine(xq,xb[i])
            dis_list.append(_dis)

        results = sorted(dis_list,reverse=True)
    top5_list.append(results[0:5])
    t2 = time.time()
    print(top5_list)
    print('faiss spend time %.4f' % ((t2 - t1) / 10))


