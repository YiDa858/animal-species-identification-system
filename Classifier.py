from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import torch
import numpy as np
from matplotlib import pyplot as plt
import cv2
import requests
import json
import os

os.environ["VISIBLE_DEVICES"] = "0"
RAW_VIT_PATH = "F:/Codefield/CODE_Python/BigDesign/models/vit-base-patch16-224"
FINETUNED_VIT_PATH = "F:/Codefield/CODE_Python/BigDesign/trained_models/vit/checkpoint-4500"
DEEPIN_VIT_PATH = "F:/Codefield/CODE_Python/BigDesign/trained_models/vit/best"


class Classifier:
    def __init__(self, inference_mode="Overall"):
        self.logits = None
        self.outputs = None
        self.inference_mode = inference_mode

        if inference_mode == "Overall":
            # 如果是针对10种动物的广度搜索
            self.model_checkpoint = FINETUNED_VIT_PATH
        elif inference_mode == "Deepin":
            # 如果是针对猫/狗的精细识别
            self.model_checkpoint = DEEPIN_VIT_PATH
        elif inference_mode == "Raw":
            # 如果是需要1000个label的原始vit
            self.model_checkpoint = RAW_VIT_PATH

        self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.model_checkpoint)
        self.model = ViTForImageClassification.from_pretrained(self.model_checkpoint)

    def inference(self, image_for_classification):
        inputs = self.feature_extractor(images=image_for_classification, return_tensors="pt")
        self.outputs = self.model(**inputs, output_attentions=True)
        self.logits = self.outputs.logits
        predicted_class_idx = self.logits.argmax(-1).item()  # 获取预测的label_id
        label_name = self.model.config.id2label[predicted_class_idx]  # 获取预测的label_name
        trust_rate = (self.logits.softmax(dim=1)[0][predicted_class_idx] * 100).item()  # 通过softmax计算可能性最高的label的置信度
        result_dict = {"label": [label_name.replace("_", " ")], "score": [round(trust_rate, 2)]}  # 创建一个字典

        return result_dict

    def visualize(self, image_for_classification):
        att_map = self.get_attention_map(image_for_classification=image_for_classification)
        self.plot_attention_map(original_img=image_for_classification, att_map=att_map)

    def get_attention_map(self, image_for_classification, get_mask=False):
        att_mat = torch.stack(self.outputs.attentions).squeeze(1)
        att_mat = torch.mean(att_mat, dim=1)
        residual_att = torch.eye(att_mat.size(1))
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
        joint_attentions = torch.zeros(aug_att_mat.size())
        joint_attentions[0] = aug_att_mat[0]
        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])
        v = joint_attentions[-1]
        grid_size = int(np.sqrt(aug_att_mat.size(-1)))
        mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
        if get_mask:
            result = cv2.resize(mask / mask.max(), image_for_classification.size)
        else:
            mask = cv2.resize(mask / mask.max(), image_for_classification.size)[..., np.newaxis]
            result = (mask * image_for_classification).astype("uint8")
        return result

    def plot_attention_map(self, original_img, att_map):
        fig, ax2 = plt.subplots(ncols=1, figsize=(16, 16))
        # ax1.set_title('Original')
        ax2.set_title('Attention Map Last Layer')
        # _ = ax1.imshow(original_img)
        _ = ax2.imshow(att_map)
        # ax1.axis('off')
        ax2.axis('off')
        plt.savefig(
            "F:\Codefield\CODE_Python\BigDesign\src\\flask\\2022_Program_Design\static\detect_object\\visualize.jpg",
            dpi=400,
            bbox_inches="tight", pad_inches=0.2)


if __name__ == "__main__":
    classifier = Classifier("Deepin")
    image = Image.open("F:\Codefield\CODE_Python\BigDesign\src\dog.jpg")
    print(classifier.inference(image_for_classification=image))
    classifier.visualize(image)
