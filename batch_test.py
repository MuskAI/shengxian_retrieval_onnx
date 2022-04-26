# from apis.infer import infer
import os, sys

sys.path.append('./apis')

from apis.construct_database import Database
from model_zoo.resnet50 import ResNetFeat
from model_zoo.mobilenet_v2_md5 import MobileNetV2Feat
from apis.infer import infer

from pprint import pprint
def inference(db, img, model_path,detector_model_path):
    '''

    :param db: saved database
    :param img: test img path
    :return: result
    '''

    method = MobileNetV2Feat(model_path=model_path,detector_model_path=detector_model_path,state='infer')
    samples = db.get_samples()
    query = method.make_single_sample(img)
    # parameters
    topd = 10
    topk = 5
    d_type = 'cosine'  # distance ty   pe  you can choose 'd1 , d2 , d3  ... d8' and 'cosine' and 'square'
    top_cls, result, std_result = infer(query, samples=samples, depth=topd, d_type=d_type, topk=topk, thr=2.0)
    # print('topk possible predicted classes:', top_cls)
    # print('topd nearest distance matching from the database:', result)
    # print('按阈值过滤后的结果:', std_result)
    return top_cls, std_result


class ErrorDigger:
    def __init__(self):
        self.error_dict1 = {'image_path': [],
                            'std_out': []}

        self.error_dict2 = {'image_path': [],
                            'std_out': []}

    def add(self, image_path, std_out, error_type=1):
        if error_type == 1:
            self.error_dict1['image_path'].append(image_path)
            self.error_dict1['std_out'].append(std_out)
        elif error_type == 2:
            self.error_dict2['image_path'].append(image_path)
            self.error_dict2['std_out'].append(std_out)

    def get_error_list(self):
        return self.error_dict1, self.error_dict2

    def __repr__(self):
        return '测试时候用于挖掘错类'


if __name__ == '__main__':

    # database is saved in ./database
    db = Database(img_dir='database', cache_dir='./cache')
    error_digger = ErrorDigger()
    project_root = os.getcwd()
    model_path = os.path.join(project_root, 'model_zoo/checkpoint/mobv2-circleloss-imagenet-e10.onnx')
    detector_model_path = os.path.join(project_root, 'model_zoo/checkpoint/pipeline-small-秤.mnn')
    db.connect_db()
    img_dir = 'database_kh_test'
    # cls_name = '菠菜'0
    cls_name_list = os.listdir(img_dir)
    print_out = []
    error_print_out = []
    right_print_out = []
    number_sum = 0
    for cls_name in cls_name_list:
        if cls_name == '.DS_Store':
            continue
        path = os.path.join(img_dir, cls_name)
        img_list = os.listdir(path)

        img_list = [os.path.join(path, name) for name in img_list]
        top_cls = []

        std_result = []
        for img_path in img_list:
            if '.DS_Store' in img_path:
                continue
            _top_cls, _std_result = inference(db, img_path, model_path,detector_model_path)
            std_result.append({'img_path': img_path, 'out': _std_result[0:5]})
            # print(_std_result)

        top1_error = []
        top2_error = []
        top3_error = []
        top4_error = []
        top5_error = []

        for i in std_result:
            img_path = i['img_path']
            i = i['out']
            print('We are testing {}'.format(cls_name))
            if i[0]['cls'] != cls_name:
                top1_error.append(i)
                error_digger.add(img_path, i, error_type=1)
                if i[1]['cls'] != cls_name:
                    top2_error.append(i)
                    error_digger.add(img_path, i, error_type=2)
                    if i[2]['cls'] != cls_name:
                        top3_error.append(i)
                        if i[3]['cls'] != cls_name:
                            top4_error.append(i)
                            if i[4]['cls'] != cls_name:
                                top4_error.append(i)
                                try:
                                    if i[5]['cls'] != cls_name:
                                        top5_error.append(i)
                                except:
                                    pass
        _ = ['类别{} Summary'.format(cls_name),
             '测试图片总数:{}'.format(len(img_list)),
             'Top1 错误数{},Top2 错误数{},Top3 错误数{},Top4 错误数{},Top5 错误数{}'.format(len(top1_error), len(top2_error),
                                                                             len(top3_error), len(top4_error),
                                                                             len(top5_error))]
        # print('类别{} Summary'.format(cls_name))
        # print('测试图片总数:{}'.format(len(img_list)))
        # print('Top1 错误数{},Top2 错误数{},Top3 错误数{}'.format(len(top1_error),len(top2_error),len(top3_error)))
        print_out.append(_)

    print('统计TOP12345 ERROR')
    for i in print_out:
        for _ in i:
            print(_)
        print()

    _top1_error,_top2_error = error_digger.get_error_list()

    pprint(_top1_error)
    pprint(_top2_error)
