

# from apis.infer import infer
from apis.dataproc_md5 import Database
from model_zoo.resnet50 import ResNetFeat
from model_zoo.mobilenet_v2_md5 import MobileNetV2Feat
from apis.infer import infer



def inference(db,img):
    '''

    :param db: saved database
    :param img: test img path
    :return: result
    '''

    method = MobileNetV2Feat()
    samples = method.make_samples(db)
    print('The length of samples:',len(samples))
    query = method.make_single_sample(img)
    print('img to be tested:', query)
    # parameters
    topd = 10
    topk = 3
    d_type = 'd1' # distance type  you can choose 'd1 , d2 , d3  ... d8' and 'cosine' and 'square'
    top_cls, result,std_result = infer(query, samples=samples, depth=topd, d_type=d_type, topk=topk,thr=0.6)
    print('topk possible predicted classes:', top_cls)
    print('topd nearest distance matching from the database:', result)
    print('按阈值过滤后的结果:', std_result)


if __name__ == '__main__':
    # database is saved in ./database
    db = Database(DB_dir='./database',DB_csv='./data.csv')
    img = './test.png'
    print("DataBase length:", len(db))
    print("DataBase classes:", db.get_class())
    inference(db, img)

