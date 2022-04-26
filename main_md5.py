# from apis.infer import infer
from apis.construct_database import Database
from model_zoo.resnet50 import ResNetFeat
from model_zoo.mobilenet_v2_md5 import MobileNetV2Feat
from apis.infer import infer

import os

def inference(db, img, model_path,detector_model_path):
    '''

    :param db: saved database
    :param img: test img path
    :return: result
    '''

    method = MobileNetV2Feat(model_path=model_path,detector_model_path=detector_model_path,state='infer')
    samples = db.get_samples()
    index = db.get_index()
    query = method.make_single_sample(img)
    # parameters
    topd = 10
    topk = 5
    d_type = 'cosine'  # distance ty   pe  you can choose 'd1 , d2 , d3  ... d8' and 'cosine' and 'square'
    top_cls, result, std_result = infer(query, index=index,samples=samples, depth=topd, d_type=d_type, topk=topk, thr=2.0)
    print('topk possible predicted classes:', top_cls)
    print('topd nearest distance matching from the database:', result)
    print('按阈值过滤后的结果:', std_result)
    return top_cls, std_result


if __name__ == '__main__':

    # database is saved in ./database
    project_root = os.getcwd()
    model_path = os.path.join(project_root, 'model_zoo/checkpoint/mobv2-circleloss-imagenet-e10.onnx')
    # detector_model_path = os.path.join(project_root, 'model_zoo/checkpoint/pipeline-small-秤.mnn')
    detector_model_path = os.path.join('/Users/musk/Desktop/实习生工作/生鲜-第二次交付-加入目标检测/pipeline-small-秤.mnn')
    db = Database(img_dir='database', cache_dir='./cache')
    db.connect_db()
    # img = '/Users/musk/Desktop/实习生工作/COCO-Hand/COCO-Hand-S/COCO-Hand-S_Images/000000000459.jpg'
    # img = '/Users/musk/Desktop/实习生工作/dataset/cargoboat_split/train_img/ship16.jpg'

    img = '/Users/musk/PycharmProjects/shengxian_retrieval_onnx/database_kh_test/丰水梨/0-md5-3bb68d396f9084181053ab1ef7f4ecc9.jpg'
    inference(db, img,model_path,detector_model_path)