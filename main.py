import argparse
import sys
import cv2
import torch
import numpy as np
import os
import gdown
import torchvision as tv
import time


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Класс хранит модели и выдает предсказания по изображениям
class Predictor:
    def __init__(self, retina, maxvit):
        self.retina = retina
        self.maxvit = maxvit

    # Функция для предсказаний
    def predict(self, orig_img):
        copy_img = orig_img.copy()
        img_width = orig_img.shape[1]
        img_height = orig_img.shape[0]
        # Изображение к диапазону от 0 до 1.
        img = orig_img.astype(np.float32) / 255.0
        # Уменьшаем
        img = cv2.resize(img, (380, 380), interpolation=cv2.INTER_AREA)
        # Располагаем каналы в порядке принятом в pyTorch
        img = img.transpose((2, 0, 1))
        t_img = torch.from_numpy(img).to(device)
        # Три класса для положения головы
        classes = ["face", "side_left", "side_right"]
        res = {"face": [], "side_left": [], "side_right": []}
        with torch.no_grad():
            predict = self.retina([t_img])
            boxes = predict[0]["boxes"]
            # Для каждой распознанной рамки вырезаем лицо
            for box in boxes:
                xmin = int(box[0] / 380 * img_width)
                xmax = int(box[2] / 380 * img_width)
                ymin = int(box[1] / 380 * img_height)
                ymax = int(box[3] / 380 * img_height)

                face_img = copy_img[ymin:ymax, xmin:xmax]
                copy_face_img = face_img.copy()
                face_img = face_img.astype(np.float32) / 255.0
                # MaxVit работает только с размером 224x224
                face_img = cv2.resize(
                    face_img, (224, 224), interpolation=cv2.INTER_AREA
                )
                face_img = face_img.transpose((2, 0, 1))
                t_face_img = torch.from_numpy(face_img).to(device)
                # Добавляем входному тензору + 1 измерение
                t_face_img = torch.unsqueeze(t_face_img, 0)
                pred = self.maxvit(t_face_img)
                class_index = pred[0].argmax().item()
                res[classes[class_index]].append(copy_face_img)
        return res


def get_predictions(src, retina, maxvit, target_dir, first, is_face, is_left, is_right):
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    if is_face and not os.path.exists(os.path.join(target_dir, "face")):
        os.mkdir(os.path.join(target_dir, "face"))
    if is_left and not os.path.exists(os.path.join(target_dir, "left")):
        os.mkdir(os.path.join(target_dir, "left"))
    if is_right and not os.path.exists(os.path.join(target_dir, "right")):
        os.mkdir(os.path.join(target_dir, "right"))
    predictor = Predictor(retina, maxvit)
    cap = cv2.VideoCapture(src)
    counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        orig_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pred = predictor.predict(orig_img)
        if (
            is_face
            and pred["face"]
            or is_left
            and pred["side_left"]
            or is_right
            and pred["side_right"]
        ):
            counter += 1
        if is_face:
            for face in pred["face"]:
                bgr_face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    os.path.join(
                        target_dir,
                        "face",
                        str(time.time()).replace(".", "") + ".png",
                    ),
                    bgr_face,
                )
        if is_left:
            for face in pred["side_left"]:
                bgr_face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    os.path.join(
                        target_dir,
                        "left",
                        str(time.time()).replace(".", "") + ".png",
                    ),
                    bgr_face,
                )
        if is_right:
            for face in pred["side_right"]:
                bgr_face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    os.path.join(
                        target_dir,
                        "right",
                        str(time.time()).replace(".", "") + ".png",
                    ),
                    bgr_face,
                )
        if counter == first:
            break
        if cv2.waitKey(1) == ord("q"):
            break
    cap.release()


def get_models():
    is_retina, is_maxvit = os.path.isfile(
        "models/best_retina2_model.pth"
    ), os.path.isfile("models/best_maxvit2_model.pth")
    if not (is_retina and is_maxvit):
        gdown.download_folder(
            "https://drive.google.com/drive/folders/1vzL9LHiGx8ogOiMKz2zeOlZlpmqO-9iQ?usp=sharing",
            output="models",
        )
    if os.path.isfile("models/best_retina2_model.pth") and os.path.isfile(
        "models/best_maxvit2_model.pth"
    ):
        retina = tv.models.detection.retinanet_resnet50_fpn_v2(
            num_classes=2,
            weights_backbone=tv.models.ResNet50_Weights.DEFAULT,
            trainable_backbone_layers=5,
            nms_thresh=0.5,
            score_thresh=0.5,
        )
        retina_checkpoint = torch.load(
            "models/best_retina2_model.pth", map_location=device, weights_only=True
        )
        retina.load_state_dict(retina_checkpoint["model_state_dict"])
        retina.to(device).eval()
        maxvit = tv.models.maxvit_t(weights=tv.models.MaxVit_T_Weights.DEFAULT)
        maxvit.classifier[-1] = torch.nn.Linear(512, 3, bias=False)
        maxvit_checkpoint = torch.load(
            "models/best_maxvit2_model.pth", map_location=device, weights_only=True
        )
        maxvit.load_state_dict(maxvit_checkpoint["model_state_dict"])
        maxvit = maxvit.to(device).eval()
        return retina, maxvit
    else:
        sys.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pp-2024-test")
    parser.add_argument("src", type=str, help="rstp url or video file")
    parser.add_argument("target", type=str, help="target dir")
    parser.add_argument(
        "--first", type=int, default=0, help="num of first positive frames"
    )
    parser.add_argument("--face", action="store_true", help="include front faces")
    parser.add_argument("--left", action="store_true", help="include side left faces")
    parser.add_argument("--right", action="store_true", help="include side right faces")
    args = parser.parse_args()
    if not (args.face or args.left or args.right):
        print("nothing to detect")
        sys.exit()
    retina, maxvit = get_models()
    get_predictions(
        args.src,
        retina,
        maxvit,
        args.target,
        args.first,
        args.face,
        args.left,
        args.right,
    )
