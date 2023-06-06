import os
import shutil

import numpy as np
import pandas as pd


def preprocess_df(raw_df):
    df = raw_df.copy()
    image_file_list = []
    image_color_list = []
    image_shape_list = []
    for i in range(len(df)):
        record_ = df.iloc[i]
        image_path = record_['image_path']
        long_dir, image_file = os.path.split(image_path)
        long_dir, image_shape = os.path.split(long_dir)
        long_dir, image_color = os.path.split(long_dir)
        image_file_list.append(image_file)
        image_shape_list.append(image_shape)
        image_color_list.append(image_color)
    df['color'] = image_color_list
    df['shape'] = image_shape_list
    df['file'] = image_file_list
    return df


def metric_analysis(gt_path, dt_path):
    gt = pd.read_csv(gt_path, delimiter='\t', names=['image_path', 'lp_num'])
    dt = pd.read_csv(dt_path, delimiter='\t', names=['image_path', 'lp_num'])

    gt = preprocess_df(gt)
    dt = preprocess_df(dt)

    dt['gt'] = gt['lp_num']
    dt['validate'] = dt['gt'] == dt['lp_num']

    total_count = 0
    shape_count = {}
    color_count = {}

    for color in np.unique(dt['color']):
        color_bool = dt['color']==color
        c_count = 0
        for shape in np.unique(dt['shape']):
            shape_bool = dt['shape']==shape
            select_dt = dt[color_bool & shape_bool]
            num_tp = sum(select_dt['validate'])
            print("Folder {} - {}: {:.4f}".format(color, shape, num_tp / len(select_dt)))
            if shape_count.get(shape, None) is None:
                shape_count[shape] = num_tp
            else:
                shape_count[shape] += num_tp
            c_count += num_tp
            total_count += num_tp
        if color_count.get(color, None) is None:
            color_count[color] = c_count
        else:
            color_count[color] += c_count
    print('-----')
    for key in color_count.keys():
        print('color {}: {:.4f}'.format(key, color_count[key] / sum(dt['color']==key)))
    print('-----')
    for key in shape_count.keys():
        print('shape {}: {:.4f}'.format(key, shape_count[key] / sum(dt['shape']==key)))
    print('-----')
    print('Total: {:.4f}'.format(total_count / len(dt)))


def compare_model(gt_path, dt_path_1, dt_path_2, dst_folder = 'compare_model'):
    dataset_dir = os.path.split(gt_path)[0]
    gt = pd.read_csv(gt_path, delimiter='\t', names=['image_path', 'lp_num'])
    dt1 = pd.read_csv(dt_path_1, delimiter='\t', names=['image_path', 'lp_num'])
    dt2 = pd.read_csv(dt_path_2, delimiter='\t', names=['image_path', 'lp_num'])

    gt = preprocess_df(gt)
    dt1 = preprocess_df(dt1)
    dt2 = preprocess_df(dt2)

    gt['pred1'] = dt1['lp_num']
    gt['pred2'] = dt2['lp_num']

    gt['validate1'] = gt['pred1'] == gt['lp_num']
    gt['validate2'] = gt['pred2'] == gt['lp_num']

    for model_1_type in [True, False]:
        for model_2_type in [True, False]:
            select_df = gt[
                (gt['validate1'] == model_1_type) & (gt['validate2'] == model_2_type)
            ]
            dir_name = ('{}_{}_{}_{}'.format(
                os.path.splitext(os.path.split(dt_path_1)[-1])[0].replace('_', ''),
                str(model_1_type).lower(),
                os.path.splitext(os.path.split(dt_path_2)[-1])[0].replace('_', ''),
                str(model_2_type).lower()
            ))
            dst_dir = os.path.join(dst_folder, dir_name)
            if os.path.isdir(dst_dir) is False:
                os.makedirs(dst_dir)

            for i in range(len(select_df)):
                record_ = select_df.iloc[i]
                src_image_path = record_['image_path']
                src_image_file = record_['file']
                src_image_name, src_image_ext = os.path.splitext(src_image_file)

                dst_image_file = '{}_{}_{}_{}_{}{}'.format(
                    record_['color'],
                    record_['shape'],
                    src_image_name,
                    record_['pred1'],
                    record_['pred2'],
                    src_image_ext
                )
                shutil.copy(
                    os.path.join(dataset_dir, src_image_path),
                    os.path.join(dst_dir, dst_image_file)
                )
