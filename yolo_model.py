import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Tesseract 설치 경로
class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.imgs = [img for img in os.listdir(img_dir) if img.endswith(".jpg")]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        label_path = os.path.join(self.label_dir, self.imgs[idx].replace(".jpg", ".txt"))

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0

        boxes = []
        labels = []

        with open(label_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                values = line.strip().split()
                if len(values) == 5:
                    class_id, x_center, y_center, width, height = map(float, values)
                    x_center, y_center, width, height = x_center * img.shape[1], y_center * img.shape[0], width * \
                                                        img.shape[1], height * img.shape[0]
                    x_min = x_center - width / 2
                    y_min = y_center - height / 2
                    x_max = x_center + width / 2
                    y_max = y_center + height / 2
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(int(class_id))
                else:
                    print(f"Unexpected label format in file {label_path}: {line.strip()}")

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx], dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transform:
            img = self.transform(img)

        return img, target


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

def test_model(model, test_loader, device):
    model.to(device)
    model.eval()

    for images, targets in test_loader:
        images = list(image.to(device).float() for image in images)  # float()로 변환
        outputs = model(images)

        for i, image in enumerate(images):
            img = image.permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype(np.uint8)  # 이미지를 uint8로 변환
            boxes = outputs[i]['boxes'].cpu().detach().numpy()
            scores = outputs[i]['scores'].cpu().detach().numpy()
            labels = outputs[i]['labels'].cpu().detach().numpy()

            filtered_boxes = boxes[scores > 0.5]
            license_texts = extract_license_plate_text(img, filtered_boxes)

            print(f"Detected license plates: {license_texts}")

            # 결과 시각화 (선택사항)
            plt.imshow(img)
            ax = plt.gca()
            for box in filtered_boxes:
                x_min, y_min, x_max, y_max = map(int, box)
                rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, color='red', linewidth=2)
                ax.add_patch(rect)
            plt.show()

def main():
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = YOLODataset(img_dir='yolov9/data/train/images', label_dir='yolov9/data/train/labels', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0,
                              collate_fn=lambda x: tuple(zip(*x)))

    num_classes = 2  # 배경 + 번호판
    model = get_model(num_classes)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        i = 0
        for images, targets in train_loader:
            images = list(image.to(device).float() for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch: {epoch}, Step: {i}, Loss: {losses.item()}")
            i += 1

        lr_scheduler.step()

    print("Training complete!")

    # 모델 저장
    torch.save(model.state_dict(), 'yolov9/models_result/model')

if __name__ == '__main__':
    main()