# from PIL import Image
import os
import re
from PIL import Image
from flask import *
from flask_cors import CORS

from Classifier import Classifier
from Detector import Detector
from imgHash import Run_threadpool
from visitBaidu import visitBaidu

basedir = os.path.abspath(os.path.dirname(__file__))
overall_classifier = Classifier("Overall")
deepin_classifier = Classifier("Deepin")
detector = Detector()

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

app.config.update(
    UPLOADED_PATH=os.path.join(basedir, 'static/detect_object/uploads'),
)

ALLOW_EXTENSIONS = ['png', 'jpg', 'jpeg']
CATS_LABEL = ["Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
              "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue", "Siamese", "Sphynx"]

DOGS_LABEL = ['miniature_pinscher', 'newfoundland', 'pomeranian', 'japanese_chin', 'yorkshire_terrier',
              'chihuahua', 'american_pit_bull_terrier', 'wheaten_terrier', 'staffordshire_bull_terrier', 'basset_hound',
              'samoyed', 'saint_bernard', 'english_setter', 'beagle', 'shiba_inu', 'great_pyrenees',
              'german_shorthaired',
              'scottish_terrier', 'leonberger', 'pug', 'english_cocker_spaniel', 'boxer', 'havanese',
              'american_bulldog', 'keeshond']

rulelist = [
    '$_GET', '$_POST', '$_REQUEST', '<?', 'php', 'eval', 'assert', 'os', 'exec', 'shell', 'base64', 'include',
    'require', 'include_once', '$_COOKIES'
]


def allowed_file(file, filename):
    file_str = file.read()
    file_str = str(file_str)
    flag = -1
    for rule in rulelist:
        flag = file_str.find(rule)
        if flag:
            break
    return ('.' in filename and filename.rsplit('.', 1)[-1] in ALLOW_EXTENSIONS) and (flag == -1)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'GET':
        return render_template('search.html')
    elif request.method == 'POST':
        key = request.form['search_input']
        print(key)
        spider = visitBaidu()
        FileList = spider.getFileList(key)
        return render_template('search.html', FileList=FileList)


@app.route('/visit', methods=['GET', 'POST'])
def visit():
    if request.method == 'POST':
        uid = request.form['id']
        print(uid)
        img_path = 'F:/Codefield/CODE_Python/BigDesign/src/flask/2022_Program_Design/static/datasets/animal-10/raw-img/' + uid
        rootdir = os.getcwd()
        files = os.listdir(img_path)
        FileList = []
        count = 0

        for name in files:
            img_path = img_path.replace('\\', "/")
            rootdir = rootdir.replace('\\', '/')
            img_path = img_path.replace(rootdir, ".")
            count = count + 1
            if os.path.isfile(img_path + '/' + name):
                FileList.append(img_path + '/' + name)
            if count == 30:
                break

        return render_template('visit.html', FileList=FileList, type=uid)
    elif request.method == 'GET':
        img_path = 'F:/Codefield/CODE_Python/BigDesign/src/flask/2022_Program_Design/static/datasets/animal-10/raw-img/cane'
        rootdir = os.getcwd()
        files = os.listdir(img_path)
        FileList = []
        count = 0

        for name in files:
            img_path = img_path.replace('\\', "/")
            rootdir = rootdir.replace('\\', '/')
            img_path = img_path.replace(rootdir, ".")
            count = count + 1
            if os.path.isfile(img_path + '/' + name):
                FileList.append(img_path + '/' + name)
            if count == 30:
                break
        return render_template('visit.html', FileList=FileList, type='cane')


@app.route('/upload')
def upload():
    return render_template('upload.html')


@app.route('/upload_file/<int:flag><int:multi_flag>', methods=['POST'])
def receive_image(flag, multi_flag):
    # print("here")
    # print(multi_flag)
    #   粗略多动物是0 1，粗略单动物是0 0
    #   精细多动物是1 1，精细单动物是1 0
    result_dict = None
    datas = None
    if request.method == 'POST':
        file = request.files['image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file, file.filename):
            file_name = file.filename
            if file_name[:-4:-1] == 'gnp':
                file_name = file_name[:-3] + 'jpg'
                image = Image.open(file.stream)
                image = image.convert("RGB")
            else:
                image = Image.open(file.stream)
            image.save(os.path.join(app.config['UPLOADED_PATH'], file_name))

            if multi_flag == 1:
                message = "Feature map"
                result_dict = {
                    "label": [],
                    "score": []
                }
                detector_result = detector.inference(image_for_detection=image, multi_animal=True)
                detector.visualize(image_for_detection=image)
                if flag == 0:
                    #   粗略多动物
                    if detector_result["multi_label_flag"] == -1:
                        pass
                    else:
                        bboxes_list = detector_result["bboxes"]
                        if len(bboxes_list) > 4:
                            bboxes_list = bboxes_list[0:4]
                        for box in bboxes_list:
                            sub_pic = image.crop(box)
                            result = overall_classifier.inference(sub_pic)
                            result_dict["label"].append(result["label"][0])
                            result_dict["score"].append(result["score"][0])
                elif flag == 1:
                    #   精细多动物
                    if detector_result["multi_label_flag"] == -1:
                        pass
                    else:
                        bboxes_list = detector_result["bboxes"]
                        if len(bboxes_list) > 4:
                            bboxes_list = bboxes_list[0:4]
                        for box in bboxes_list:
                            sub_pic = image.crop(box)
                            result = deepin_classifier.inference(sub_pic)
                            result_dict["label"].append(result["label"][0])
                            result_dict["score"].append(result["score"][0])
            elif multi_flag == 0:
                message = "Attention map"
                if flag == 0:
                    #   粗略单动物
                    result_dict = overall_classifier.inference(image_for_classification=image)
                    overall_classifier.visualize(image_for_classification=image)
                    datas = Run_threadpool(os.path.join(app.config['UPLOADED_PATH'], file_name),
                                           result_dict["label"][0])
                if flag == 1:
                    #   精细单动物
                    result_dict = deepin_classifier.inference(image_for_classification=image)
                    deepin_classifier.visualize(image_for_classification=image)
                    datas = Run_threadpool(os.path.join(app.config['UPLOADED_PATH'], file_name),
                                           "cane" if result_dict["label"][0] in DOGS_LABEL else "gatto")
        else:
            return render_template('upload.html', not_allowed=True)
        if multi_flag == 1:
            raw_image = "/static/detect_object/result.jpg"
        else:
            raw_image = f"/static/detect_object/uploads/{file_name}"
        visualize_image = "/static/detect_object/visualize.jpg"

        return render_template('upload.html', result_dict=result_dict, datas=datas,
                               multiple_label_flag=multi_flag,
                               visualize_image=visualize_image,
                               raw_image=raw_image,
                               index_list=[1, 2, 3, 4],
                               message=message)


if __name__ == '__main__':
    app.run()
