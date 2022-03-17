#__version__= demo_beta_v_1_0
#__author__ = haolin
import numpy as np
import onnxruntime as ort
import cv2
import os
import tqdm
from six.moves import cPickle

RES_model = 'resnet50'
pick_layer = 'avgPooling'
d_type = 'd1'
cache_dir = 'cache'
depth = 3  # retrieved depth, set to None will count the ap for whole database

if not os.path.exists(cache_dir):
  os.makedirs(cache_dir)

class ResNetFeat(object):
    def make_single_sample(self, d_img, verbose=True):

        # res_model = ResidualNet(model=RES_model)
        res_model = ort.InferenceSession("./model_zoo/checkpoint/ret_resnet50.onnx")
        input_name = res_model.get_inputs()[0].name
        output_name = res_model.get_outputs()[0].name
        result = None
        print("Start to use ResNet 50 to generate features for DataBase")

        img =cv2.imread(d_img)
        img = img.astype(np.float32)
        mean = np.array([0.0, 0.0, 0.0])
        std = np.array([256.0, 256.0, 256.0])
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
            'hist': d_hist}
        return result

    def make_samples(self, db, verbose=True):
        sample_cache = '{}-{}'.format(RES_model, pick_layer)

        try:
            samples = cPickle.load(open(os.path.join(cache_dir, sample_cache), "rb", True))
            for sample in samples:
                sample['hist'] /= np.sum(sample['hist'])  # normalize
            if verbose:
                print("Using cache..., config=%s, distance=%s, depth=%s" % (sample_cache, d_type, depth))
        except:
            if verbose:
                print("Counting histogram..., config=%s, distance=%s, depth=%s" % (sample_cache, d_type, depth))

            # res_model = ResidualNet(model=RES_model)
            res_model = ort.InferenceSession("./model_zoo/checkpoint/ret_resnet50.onnx")
            input_name = res_model.get_inputs()[0].name
            output_name = res_model.get_outputs()[0].name
            samples = []
            data = db.get_data()
            print("Start to use ResNet 50 to generate features for DataBase")
            for d in tqdm.tqdm(data.itertuples()):
                d_img, d_cls = getattr(d, "img"), getattr(d, "cls")
                img =cv2.imread(d_img)
                img = img.astype(np.float32)
                mean = np.array([0.0, 0.0, 0.0])
                std = np.array([256.0, 256.0, 256.0])
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
                    'hist': d_hist})
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
            cPickle.dump(samples, open(os.path.join(cache_dir, sample_cache), "wb", True))
            print("Finish features generation for DataBase")
        return samples