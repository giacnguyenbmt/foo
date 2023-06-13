import os
import time
import sys

import cv2
import numpy as np
import onnxruntime

from metric import metric_analysis


class LPR:
    def __init__(self, model_file, chars, size) -> None:
        self.model_file = model_file
        self.chars = chars
        self.img_w = size[0]
        self.img_h = size[1]

        assert self.model_file is not None
        assert os.path.exists(self.model_file)
        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        self.session = onnxruntime.InferenceSession(self.model_file, so)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = self.session.get_outputs()[0].name

    def preprocess(self, image, split_ratio=1.05):
        # image = self.split_and_concate_w_ratio(image, split_ratio)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_w, self.img_h))
        image = image.astype('float32')
        image -= 127.5
        image *= 0.0078125
        image = np.transpose(image, (2, 0, 1))
        img_batch = np.expand_dims(image, axis=0)
        return img_batch

    def forward(self, img_batch):
        preds = self.session.run([self.output_names], {self.input_name: img_batch})
        return preds
    
    def greedy_decode(self, preds):
        preds = preds[0].transpose((0, 2, 1))  # => N, C, T
        for i in range(preds.shape[0]):
            pred = preds[i, :, :]
            pred_label = list()
            for j in range(pred.shape[1]):
                pred_label.append(np.argmax(pred[:, j], axis=0))
            no_repeat_blank_label = list()
            pre_c = pred_label[0]
            if pre_c != len(self.chars) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in pred_label: # dropout repeate label and blank label
                if (pre_c == c) or (c == len(self.chars) - 1):
                    if c == len(self.chars) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            lb = ""
            for i in no_repeat_blank_label:
                if i<len(self.chars):
                    lb += self.chars[i]
        return lb

    def recognize(self, 
                  image, 
                  use_preprocess=True):
        if use_preprocess:
            image = self.preprocess(image)
        preds = self.forward(image)
        chars = self.greedy_decode(preds)
        return chars
    
    def split_and_concate_w_ratio(self, raw_img, ratio=1.05):
        i_h, i_w, _ = raw_img.shape
        if i_w / i_h < 2.5:
            img = raw_img.copy()
            # Nếu chiều cao là số lẻ, padding thêm 1 hàng vào cuối ảnh
            if img.shape[0] % 2 == 1:
                img = np.concatenate((img, img[-1:]), axis=0)
            top_limit = int((img.shape[0] // 2) * ratio)
            bottom_limit = img.shape[0] - top_limit
            img = np.concatenate((img[:top_limit], img[bottom_limit:]), axis=1)
            return img
        else:
            return raw_img


def test_dataset(lpr, dataset_dir, output_dir='output'):
    if dataset_dir.endswith('/'):
        dataset_dir = dataset_dir[:-1]
    if os.path.isdir(output_dir) is False:
        os.makedirs(output_dir)

    data_name = os.path.split(dataset_dir)[-1]
    input_list_file = os.path.join(dataset_dir, 'rec.txt')
    dt_path = os.path.join(output_dir, data_name + '.txt')
    print(dt_path)

    with open(input_list_file) as f:
        input_content = f.readlines()

    img_list = []
    valid_image_file_list = []
    gt_list = []
    counter = 0

    for line in input_content:
        if line == '\n':
            continue
        line_info = line.strip().split('\t')
        gt_list.append(line_info[1])
        img_path = os.path.join(dataset_dir, line_info[0])
        img = cv2.imread(img_path)
        img_list.append(lpr.preprocess(img))
        valid_image_file_list.append(line_info[0])
    st = time.time()
    with open(dt_path, 'w') as f:
        for i in range(len(valid_image_file_list)):
            chars = lpr.recognize(img_list[i], 
                                  use_preprocess=False)
            if chars == gt_list[i]:
                counter += 1
            f.write('{}\t{}\n'.format(
                valid_image_file_list[i],
                chars
            ))

    print('FPS:', len(img_list) / (time.time() - st))
    metric_analysis(input_list_file, dt_path)


if __name__ == '__main__':
    model_path = sys.argv[1]
    dataset_dir = sys.argv[2]

    model_name = os.path.split(os.path.splitext(model_path)[0])[-1]
    output_dir = os.path.join('output', model_name)
    lpr = LPR(model_file=model_path, chars='0123456789ABCDĐEFGHKLMNPQRSTUVXYZ ', size=(188, 24))

    test_dataset(lpr, dataset_dir, output_dir)
