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

    def make_single_sample(self, d_img, verbose=True):
        # res_model = ResidualNet(model=RES_model)
        res_model = ort.InferenceSession(self.model_path)
        input_name = res_model.get_inputs()[0].name
        output_name = res_model.get_outputs()[0].name
        result = None
        print("Start to use MobileNet V2 to generate features for DataBase")
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
            'md5': md5_code,
        }
        return result

    def make_samples(self, db, verbose=True):
        sample_cache = '{}-{}-{}'.format(self.RES_model, self.pick_layer, 'md5')

        try:
            # 如果有数据库文件直接load
            samples = cPickle.load(open(os.path.join(self.cache_dir, sample_cache), "rb", True))
            # 增加和删除检查
            diffs = self.check_duplicate(samples, db)

            # 如果有需要增加到数据库的
            if diffs['insert']:
                for idx, item in enumerate(diffs['insert']):
                    hist = self.make_single_sample(d_img=item['img'].replace('../', './'))['hist']
                    samples.append(
                        {'img': item['img'],
                         'cls': item['cls'],
                         'hist': hist,
                         'md5': item['md5']}
                    )
            # 如果有需要从数据库中删除的
            if diffs['del']:
                for i in diffs['del']:
                    samples.remove(i)

            # 如果数据库有修改，则重新保存
            if diffs['del'] != [] or diffs['insert'] != []:
                cPickle.dump(samples, open(os.path.join(self.cache_dir, sample_cache), "wb", True))

            for sample in samples:
                sample['hist'] /= np.sum(sample['hist'])  # normalize
            if verbose:
                print("Using cache..., config=%s, distance=%s, depth=%s" % (sample_cache, self.d_type, self.depth))

            return samples
        except:
            # 如果没有数据库文件则需要推理生成
            # 第一次是没有数据库文件的，后面的数据库文件只是做增删项操作，不会直接删除数据库文件
            if verbose:
                print(
                    "Counting histogram..., config=%s, distance=%s, depth=%s" % (sample_cache, self.d_type, self.depth))

            # res_model = ResidualNet(model=RES_model)
            res_model = ort.InferenceSession(self.model_path)
            input_name = res_model.get_inputs()[0].name
            output_name = res_model.get_outputs()[0].name
            samples = []
            data = db.get_data()
            print("Start to use MobileNet V2 to generate features for DataBase")
            for d in tqdm.tqdm(data.itertuples()):
                d_img, d_cls = getattr(d, "img"), getattr(d, "cls")
                d_img = d_img.replace('../', './')
                # 对读取的图片进行编码
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
                samples.append({
                    'img': d_img,
                    'cls': d_cls,
                    'hist': d_hist,
                    'md5': md5_code,
                })
                # img = scipy.misc.imread(d_img, mode="RGB")
                # img = img[:, :, ::-1]  # switch to BGR
                # img = np.transpose(img, (2, 0, 1)) / 255.
                # img[0] -= means[0]  # reduce B's mean
                # img[1] -= means[1]  # reduce G's mean
                # img[2] -= means[2]  # reduce R's mean
                # img = np.expand_dims(img, axis=0)
                # try:
                #     if use_gpu:
                #         inputs = torch.autograd.Variable(torch.from_numpy(img).cuda().float())
                #     else:
                #         inputs = torch.autograd.Variable(torch.from_numpy(img).float())
                #     d_hist = res_model(inputs)[pick_layer]
                #     d_hist = d_hist.data.cpu().numpy().flatten()
                #     d_hist /= np.sum(d_hist)  # normalize
                #
                #
                #     samples.append({
                #         'img': d_img,
                #         'cls': d_cls,
                #         'hist': d_hist
                #     })
                # except:
                #     pass
            cPickle.dump(samples, open(os.path.join(self.self.cache_dir, sample_cache), "wb", True))
            print("Finish features generation for DataBase")
            return samples

    def check_duplicate(self, samples, db):
        """
        通过csv与数据库中对比,找出多余的，和缺失的项，与csv保持一致

        实现逻辑:
        1. 对比文件夹目录和csv表，检查是否有增删-->更新csv文件
        2. 更具csv文件增删信息-->更新数据库文件
        3. csv 和 数据库文件依据md5编码匹配

        Args:
             samples: 数据库，是一个list
             data_csv: 与文件目录保持一致

             return  diffs = [{
            'insert': insert_list,
            'del': del_list
        }]
        """

        # 获取CSV文件
        data = db.get_data()
        print("DB length:", len(db))

        # 获取所有csv文件中的md5项
        csv_md5_list = list(data['md5'])
        csv_md5_set = set(csv_md5_list)

        samples_md5_list = [i['md5'] for i in samples]
        samples_md5_set = set(samples_md5_list)

        # 用CSV文件与数据库匹配开始
        # 求两个集合的交、差集
        c_s = list(csv_md5_set.difference(samples_md5_set))  # csv_set - samples_set 新增的数据
        s_c = list(samples_md5_set.difference(csv_md5_set))  # samples_set - csv_set 删除的数据

        # 从csv中找到需要添加的图片信息
        insert_list = []
        for i in c_s:
            insert_list.append(dict(data.loc[csv_md5_list.index(i)]))

        # 从samples中找出需要删除的信息
        del_list = []
        for i in s_c:
            del_list.append(samples[samples_md5_list.index(i)])

        diffs = {
            'insert': insert_list,
            'del': del_list
        }
        return diffs

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

    # def insert_samples(self,samples, diffs):
    #     """
    #     插入单个数据
    #     """
    #     assert isinstance(diffs,list)
    #     for idx,item in enumerate(diffs):
    #
    #
    #
    # def del_samples(self):
    #     pass
