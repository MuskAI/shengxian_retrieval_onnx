# __version__= demo_beta_v_2_0
# __author__ = haolin & haoran
import numpy as np
import onnxruntime as ort
import cv2
import os,sys
import tqdm
from six.moves import cPickle
sys.path.append('../model_zoo/')
sys.path.append('./model_zoo/')


import matplotlib.pyplot as plt
from hashlib import md5
from inference_deeptk_sx import InferenceDeepTk

class MobileNetV2Feat(object):
    def __init__(self, cache_dir='cache', model_path='./model_zoo/checkpoint/ret_mobilenet_v2.onnx'
                 , detector_model_path='../model_zoo/checkpoint/pipeline-small-秤.onnx', autocrop=True,state='insert'):
        """

        @param cache_dir:
        @param model_path:
        @param detector_model_path: 目标检测模型的路径
        @param autocrop: 是否使用目标检测模型进行AI Crop
        """
        self.RES_model = 'mobilenetV2'
        self.pick_layer = 'avgPooling'
        # self.d_type = 'd1'
        self.cache_dir = cache_dir
        self.depth = 3  # retrieved depth, set to None will count the ap for whole database
        self.model_path = model_path
        self.state = state
        # 判断模型存在情况
        if not os.path.isfile(model_path):
            raise '检索模型不存在'
        if not os.path.isfile(detector_model_path):
            raise '检测模型不存在'

        # 是否预先使用目标检测
        if autocrop:
            self.scale_detector = InferenceDeepTk(model_path=detector_model_path, input_tensor_size=(1, 3, 320, 320), classes=('d','w','c'))

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def make_single_sample(self, d_img, verbose=True,md5_encoding=True):
        # res_model = ResidualNet(model=RES_model)
        # crop_range_ratio = (0.2,0.8)
        res_model = ort.InferenceSession(self.model_path)
        input_name = res_model.get_inputs()[0].name
        output_name = res_model.get_outputs()[0].name
        result = None
        if verbose:
            print("Start to use MobileNet V2 to generate features for DataBase")
        if md5_encoding:
            md5_code = self.get_md5(img_path=d_img)

        # 读取图片
        img0 = cv2.imread(d_img)
        # 在这里进行图片的裁剪
        img = self.__get_bbox_then_crop(img0, state=self.state)

        # 如果没有检出目标则返回None

        if img is None:
            return None
        # Debug 的时候使用
        # plt.subplot(121)
        # plt.imshow(np.array(img0, dtype='uint8'))
        # plt.subplot(122)
        # plt.imshow(np.array(img, dtype='uint8'))
        # plt.savefig('/Users/musk/Desktop/实习生工作/单检测_测试_crop_expand1/{}'.format(d_img.split('/')[-1]))
        # plt.clf()
        # plt.show()
        img = img.astype(np.float32)
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        mean = np.float64(mean.reshape(1, -1))
        stdinv = 1 / np.float64(std.reshape(1, -1))
        cv2.subtract(img, mean, img)  # inplace
        cv2.multiply(img, stdinv, img)  # inplace
        img = cv2.resize(img, (224, 224))
        img = np.array([img])
        img = np.transpose(img, [0, 3, 1, 2])
        output = res_model.run([output_name], {input_name: img})
        d_hist = output[0][0]
        d_hist /= np.sum(d_hist)
        result = {
            'img': d_img,
            'cls': 'unknown',
            'hist': d_hist,
            'md5': md5_code if md5_encoding else 'unknown',
        }
        return result


    def get_md5(self, img_path):
        """
        对图片进行md5编码，编码成功则返回str，失败则返回None
        Args:
            img_path: [str] 需要编码图片的地址

        """
        # assert os.path.isfile(img_path)

        try:
            with open(img_path, 'rb') as img:
                md5_code = md5(img.read()).hexdigest()
        except Exception as e:
            return None

        return md5_code

    def __get_bbox_then_crop(self,img,expand=0.1,selected_type='area',top_k=2,state='insert'):
        """
        调用秤盘检测的模型，获取秤盘的位置，然后crop输入图片
        @param img:
        @param expand: 默认为0.1
        @param selected_type:
        @param top_k: 根据分数
        @return:
        """
        assert selected_type in ('area','score')
        assert state in ('infer','insert') # 执行状态，要区分增量学习或构建数据库与检索。
        w = img.shape[1]
        h = img.shape[0]

        # 获取目标检测的结果
        output = self.scale_detector.inference(img,nms=False)

        # 如果什么都没有检出
        if output is None:
            return None

        # 如果有检出，开始执行算法
        selected = []
        for i in output:
            # 针对mnn的bug做如下处理，如果最高分数的框小于阈值则判断为没有检出
            if i['score']<0.4:
                break

            if i['name'] == 'd':
                selected.append(i)
                if top_k == 0:
                    break
                else:
                    top_k -= 1
            else:
                pass
        if len(selected) == 0:
            # 如果一个单都没检测出来则采用人工设定框截取的方法
            crop_range_ratio_w = (0.2,0.8)
            crop_range_ratio_h = (0,1)
            # selected = output[0]
            selected ={
                'lx':0,
                'ly':0,
                'rx':w,
                'ry':h,
                'score':0,
            }
            # 如果是insert状态则返回None
            if self.state == 'insert':
                return None

            # 如果是infer状态则按照人工设定的框截取

            selected['lx'] = 0 if selected['lx'] <0 else selected['lx']
            selected['rx'] = w if selected['lx'] > w else selected['rx']
            img = img[int(selected['ly']+(selected['ry']-selected['ly']) * crop_range_ratio_h[0]):int(selected['ly']+(selected['ry']-selected['ly']) * crop_range_ratio_h[1]):,
                  int(selected['lx']+(selected['rx']-selected['lx']) * crop_range_ratio_w[0]):int(selected['lx']+(selected['rx']-selected['lx']) * crop_range_ratio_w[1])]


        elif len(selected) ==1:
            # 如果只有一个单
            selected = selected[0]
            bbox_w = selected['rx'] - selected['lx']
            bbox_h = selected['ry'] - selected['ly']

            # 对裁剪的框进行expand
            _ly = selected['ly'] - (bbox_h * expand) if (selected['ly'] - (bbox_h * expand)) > 0 else 0
            _ry = selected['ry'] + (bbox_h * expand) if (selected['ry'] + (bbox_h * expand)) < h else h
            _lx = selected['lx'] - (bbox_w * expand) if (selected['lx'] - (bbox_w * expand)) > 0 else 0
            _rx = selected['rx'] + (bbox_w * expand) if (selected['rx'] + (bbox_w * expand)) < w else w
            img = img[int(_ly):int(_ry), int(_lx):int(_rx), :]
        else:
            # 如果有多个单
            _area1 = (selected[0]['rx'] - selected[0]['lx']) * (selected[0]['ry'] - selected[0]['ly'])
            _area2 = (selected[1]['rx'] - selected[1]['lx']) * (selected[1]['ry'] - selected[1]['ly'])
            selected = selected[1] if (_area2 > _area1) and (selected[0]['score'] - selected[1]['score'] < 0.4) else selected[0]

            bbox_w = selected['rx']-selected['lx']
            bbox_h = selected['ry']-selected['ly']

            # 对裁剪的框进行expand
            _ly = selected['ly'] - (bbox_h*expand) if (selected['ly'] - (bbox_h*expand)) > 0 else 0
            _ry = selected['ry'] + (bbox_h*expand) if (selected['ry'] + (bbox_h*expand)) < h else h
            _lx = selected['lx'] - (bbox_w*expand) if (selected['lx'] - (bbox_w*expand)) > 0 else 0
            _rx = selected['rx'] + (bbox_w*expand) if (selected['rx'] + (bbox_w*expand)) < w else w
            img = img[int(_ly):int(_ry), int(_lx):int(_rx),:]
        return img
