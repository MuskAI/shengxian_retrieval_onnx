# __version__= demo_beta_v_2_0
# __author__ = haolin & haoran
import numpy as np
import onnxruntime as ort
import cv2
import os
import tqdm
from six.moves import cPickle
from apis.dataproc_md5 import Database
from hashlib import md5


class MobileNetV2Feat(object):
    def __init__(self, cache_dir='cache', model_path='./model_zoo/checkpoint/ret_mobilenet_v2.onnx'):
        self.RES_model = 'mobilenetV2'
        self.pick_layer = 'avgPooling'
        self.d_type = 'd1'
        self.cache_dir = cache_dir
        self.depth = 3  # retrieved depth, set to None will count the ap for whole database
        self.model_path = model_path
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def make_single_sample(self, d_img, verbose=True,md5_encoding=True):
        # res_model = ResidualNet(model=RES_model)
        res_model = ort.InferenceSession(self.model_path)
        input_name = res_model.get_inputs()[0].name
        output_name = res_model.get_outputs()[0].name
        result = None
        if verbose:
            print("Start to use MobileNet V2 to generate features for DataBase")
        if md5_encoding:
            md5_code = self.get_md5(img_path=d_img)

        img = cv2.imread(d_img)
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

