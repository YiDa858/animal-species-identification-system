import PIL.Image
import matplotlib.pyplot as plt
import torch
from transformers import DetrFeatureExtractor
from transformers import DetrForObjectDetection

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
DETR_PATH = "F:/Codefield/CODE_Python/BigDesign/models/detr-resnet-50"


class Detector:
    def __init__(self):
        self.bboxes_scaled = None
        self.postprocessed_outputs = None
        self.target_sizes = None
        self.keep = None
        self.probas = None
        self.logits = None
        self.model_checkpoint = DETR_PATH
        self.feature_extractor = DetrFeatureExtractor.from_pretrained(self.model_checkpoint)
        self.model = DetrForObjectDetection.from_pretrained(self.model_checkpoint)
        #   载入feature extractor和model

    def plot_results(self, pil_img, prob, boxes, result_dict):
        plt.figure(figsize=(16, 10))
        plt.imshow(pil_img)
        ax = plt.gca()
        colors = COLORS * 100
        #   画图准备

        i = 1
        label_str = ""
        score_str = ""

        for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color=c, linewidth=3))
            cl = p.argmax()
            text = f'{self.model.config.id2label[cl.item()]}: {p[cl]:0.3f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
            #   画目标检测框图

            label = f'{self.model.config.id2label[cl.item()]}'
            score = f'{p[cl]:0.3f}'
            label_str += f"({i}): {label} "
            score_str += f"({i}): {score} "
            i += 1
            #   格式化字符串以添加到result_dict中

        result_dict["label"] = label_str[0: len(label_str) - 2]
        result_dict["score"] = score_str[0: len(score_str) - 2]
        #   添加最终结果到result_dict中

        plt.axis('off')
        plt.savefig("F:\Codefield\CODE_Python\BigDesign\src\\flask\\2022_Program_Design\static\detect_object\\result.jpg", bbox_inches="tight", pad_inches=0.0)
        #   保存

        return result_dict

    def inference(self, image_for_detection, multi_animal=False):
        inputs = self.feature_extractor(images=image_for_detection, return_tensors="pt")
        outputs = self.model(**inputs)

        self.logits = outputs.logits
        #   推理结论
        self.probas = outputs.logits.softmax(-1)[0, :, :-1]
        #   概率
        self.keep = self.probas.max(-1).values > 0.9
        #   置信度大于0.9的纳入考虑
        self.target_sizes = torch.tensor(image_for_detection.size[::-1]).unsqueeze(0)
        #   获取size以供post_process方法使用
        self.postprocessed_outputs = self.feature_extractor.post_process(outputs, self.target_sizes)
        #   使用post_process获得正确的bboxes
        self.bboxes_scaled = self.postprocessed_outputs[0]['boxes'][self.keep]
        #   获取正确bboxes以用于画框图

        result_dict = {
            "label": "",
            "score": ""
        }
        #   返回的文件

        multi_label_flag = 0
        label_number = len(self.probas[self.keep])
        #   计算是否为多类动物

        if label_number > 1:
            #   如果是多类动物
            multi_label_flag = label_number
            result_dict = self.plot_results(image_for_detection, self.probas[self.keep], self.bboxes_scaled,
                                            result_dict)
            result_dict["bboxes"] = self.bboxes_scaled.tolist()
        elif label_number == 0:
            #   如果没有可供检测的目标
            multi_label_flag = -1
            result_dict["label"] = "No label found, please check the input image."
            result_dict["score"] = "No label found, please check the input image."
        else:
            #   如果只有一类动物
            cl = self.probas[self.keep][0].argmax()
            label = f'{self.model.config.id2label[cl.item()]}'
            score = f'{self.probas[self.keep][0][cl]:0.3f}'
            result_dict["label"] = f"label is {label}"
            result_dict["score"] = f"score is {score}"

        result_dict["multi_label_flag"] = multi_label_flag
        #   这个值告诉前端需要如何渲染
        return result_dict

    def visualize(self, image_for_detection):
        conv_features = []
        # use lists to store the outputs via up-values

        hooks = [
            self.model.model.backbone.conv_encoder.register_forward_hook(
                lambda self, input, output: conv_features.append(output)
            ),
        ]
        inputs = self.feature_extractor(images=image_for_detection, return_tensors="pt")

        outputs = self.model(**inputs, output_attentions=True)
        # propagate through the model

        for hook in hooks:
            hook.remove()

        conv_features = conv_features[0]
        # don't need the list anymore
        dec_attn_weights = outputs.cross_attentions[-1]
        # get cross-attention weights of last decoder layer - which is of shape (batch_size, num_heads, num_queries, width*height)
        dec_attn_weights = torch.mean(dec_attn_weights, dim=1).detach()
        # average them over the 8 heads and detach from graph

        # get the feature map shape
        h, w = conv_features[-1][0].shape[-2:]

        fig, axs = plt.subplots(ncols=len(self.keep.nonzero()), nrows=2, figsize=(10, 7))
        colors = COLORS * 100

        for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(self.keep.nonzero(), axs.T, self.bboxes_scaled):
            if len(self.bboxes_scaled) == 1:
                ax = axs.T[0]
                #   只有一个label
            else:
                ax = ax_i[0]
                #   有多个label
            ax.imshow(dec_attn_weights[0, idx].view(h, w))
            ax.axis('off')
            ax.set_title(f'query id: {idx.item()}')
            if len(self.bboxes_scaled) == 1:
                ax = axs.T[1]
                #   只有一个label
            else:
                ax = ax_i[1]
                #   有多个label
            ax.imshow(image_for_detection)
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color='blue', linewidth=3))
            ax.axis('off')
            ax.set_title(self.model.config.id2label[self.probas[idx].argmax().item()])
        fig.savefig("F:\Codefield\CODE_Python\BigDesign\src\\flask\\2022_Program_Design\static\detect_object\\visualize.jpg",
                    bbox_inches="tight")
        #   保存


if __name__ == '__main__':
    detector = Detector()
    image = PIL.Image.open("../../dog1.jpg")
    print(detector.inference(image))
    detector.visualize(image)
