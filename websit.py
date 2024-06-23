import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
import torch
import cv2
import numpy as np
import pytesseract
from torchvision import transforms, models  # torchvision.models 추가
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 파일 크기 제한

# Tesseract OCR 경로 설정 (필요한 경우)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Tesseract 설치 경로


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def get_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_model(model_path, num_classes):
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def extract_license_plate_text(image, boxes):
    license_texts = []
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        plate_img = image[y_min:y_max, x_min:x_max]
        plate_img_gray = cv2.cvtColor(plate_img, cv2.COLOR_RGB2GRAY)
        text = pytesseract.image_to_string(plate_img_gray, config='--psm 7')  # 번호판 인식을 위한 설정
        license_texts.append(text.strip())
    return license_texts


def process_image(image_path, model, device):
    transform = transforms.Compose([transforms.ToTensor()])
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img_rgb).unsqueeze(0).to(device).float()
    with torch.no_grad():
        output = model(img_tensor)

    boxes = output[0]['boxes'].cpu().detach().numpy()
    scores = output[0]['scores'].cpu().detach().numpy()
    filtered_boxes = boxes[scores > 0.5]
    license_texts = extract_license_plate_text(img_rgb, filtered_boxes)
    return license_texts


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # 파일 체크
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # 모델 로드 및 이미지 처리
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            model = load_model('yolov9/models_result/model', num_classes=2)
            model.to(device)
            license_texts = process_image(file_path, model, device)

            # 결과 표시
            return render_template('result.html', license_texts=license_texts, filename=filename)
    return render_template('upload.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)