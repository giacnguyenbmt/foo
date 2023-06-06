import os
import glob
import shutil


def split_true_false_sample(gt_path, dt_path, dst_dir):
    data_dir = os.path.split(gt_path)[0]
    true_dir = 'true_sample'
    false_dir = 'false_sample'
    with open(gt_path) as f:
        gt_content = f.readlines()
    with open(dt_path) as f:
        dt_content = f.readlines()
    for gt_line, dt_line in zip(gt_content, dt_content):
        gt_info = gt_line.strip().split('\t')
        dt_info = dt_line.strip().split('\t')
        sample_dir = true_dir if gt_info[1] == dt_info[1] else false_dir
        sub_dir, image_file = os.path.split(gt_info[0])
        image_name, image_ext = os.path.splitext(image_file)
        src_path = os.path.join(data_dir, gt_info[0])
        dst_folder_dir = os.path.join(dst_dir, sample_dir, sub_dir)
        if os.path.isdir(dst_folder_dir) is False:
            os.makedirs(dst_folder_dir)
        dst_path = os.path.join(
            dst_folder_dir,
            image_name + '_{}{}'.format(dt_info[1], image_ext)
        )
        shutil.copy(src_path, dst_path)


def split_true_false_sample_irled(gt_dir, dt_path, dst_dir):
    data_dir = gt_dir
    true_dir = 'true_sample'
    false_dir = 'false_sample'
    with open(dt_path) as f:
        dt_content = f.readlines()
    for dt_line in dt_content:
        dt_info = dt_line.strip().split('\t')
        gt_num = dt_info[0].split('-')[-1].split('.')[0]
        dt_num = dt_info[1]
        sample_dir = true_dir if gt_num == dt_num else false_dir
        sub_dir, image_file = os.path.split(dt_info[0])
        print(dt_info[0])
        image_name, image_ext = os.path.splitext(image_file)
        src_path = os.path.join(data_dir, dt_info[0])
        dst_folder_dir = os.path.join(dst_dir, sample_dir, sub_dir)
        print(dst_folder_dir)
        print(sub_dir)
        return
        if os.path.isdir(dst_folder_dir) is False:
            os.makedirs(dst_folder_dir)
        dst_path = os.path.join(
            dst_folder_dir,
            image_name + '_{}{}'.format(dt_num, image_ext)
        )
        # shutil.copy(src_path, dst_path)


if __name__ == '__main__':
    gt_path = '/home/nguyenpdg/Documents/GiacNguyen/lpr/data/ppocr-format/pp_vn_real_1row_lp_total/rec.txt'
    dt_path = 'lprnet_v5.1.4_pp_vn_real_1row_lp_total.txt'
    dst_dir = 'data_analysis/pp_vn_real_1row_lp_total'
    split_true_false_sample(gt_path, dt_path, dst_dir)


    gt_dir = '/home/nguyenpdg/Documents/GiacNguyen/lpr/label_LP/label_by_model/data/irled_data/rec.txt'
    dt_path = 'lprnet_v5.1.4_irled_data.txt'
    dst_dir = 'data_analysis/irled_data'
    split_true_false_sample(gt_dir, dt_path, dst_dir)
