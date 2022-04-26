import numpy as np
from scipy import spatial
import faiss

def infer(query,samples=None,index=None, depth=None, topk=5, thr=0.4):
    assert index is not None, "need to give either samples"
    q_img, q_cls, q_hist = query['img'], query['cls'], query['hist']
    # 利用faiss进行检索
    D, I = index.search(-q_hist, topk)
    # 标准化输出
    std_results = []
    top_cls = []
    for _dis,_i in zip(D,I):
        if samples[_i]['cls'] not in top_cls:
            top_cls.append(results[i]['cls'])
            std_results.append({'cls': results[i]['cls'],
                                'dis': results[i]['dis']})
        if len(top_cls) >= topk:
            break

    if depth and depth <= len(results):
        results = results[:depth]



    return top_cls, results, std_results
