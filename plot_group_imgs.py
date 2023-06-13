import glob
import os
import random

import cv2
import numpy as np


def plot_group(file_template, num_h, num_w, show_img=False):
    img_path_list = glob.glob(file_template)
    random.shuffle(img_path_list)
    select_list = list()

    mean_w = 0
    mean_h = 0

    counter = 0
    idx = -1
    while counter < num_h * num_w:
        idx += 1
        img_path = img_path_list[idx]
        img = cv2.imread(img_path)
        h, w, c = img.shape
        if w / h > 2.5:
            continue
        else:
            select_list.append(img)
            counter += 1
            mean_w += w
            mean_h += h
    mean_w = round(mean_w / counter)
    mean_h = round(mean_h / counter)

    processed_list = [cv2.resize(img, (mean_w, mean_h)) for img in select_list]
    print(len(processed_list))

    verticle_list = list()
    for i in range(num_h):
        verticle_list.append(np.concatenate(processed_list[i*num_w: (i+1)*num_w], axis=1))
    final_img = np.concatenate(verticle_list, axis=0)
    if show_img:
        cv2.imshow('img', final_img)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return final_img


if __name__ == '__main__':
    illegible_template = os.path.join('data/its_cam_s3_analysis/illegible', '*/*.jpg')
    illegible_img = plot_group(illegible_template, 10, 10)
    cv2.imwrite('illegible_10x10.jpg', illegible_img)
    # legible_template = os.path.join('data/its_cam_s3_analysis/positive', '*/*.jpg')
    # legible_img = plot_group(legible_template, 3, 4)
    # cv2.imwrite('legible_img.jpg', legible_img)
    illegible_template = os.path.join('data/irled_data/false', '*.jpg')
    illegible_img = plot_group(illegible_template, 8, 6)
    cv2.imwrite('data/irled_data/false_img.jpg', illegible_img)
    legible_template = os.path.join('data/irled_data/true', '*.jpg')
    legible_img = plot_group(legible_template, 8, 6)
    cv2.imwrite('data/irled_data/true_img.jpg', legible_img)
