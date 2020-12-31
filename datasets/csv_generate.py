import os
import shutil
import re
import pickle
import csv
import PIL.Image as Image
import numpy as np

"""
miniImageNet structure:
miniImageNet
    class_dict
            label:image_index
    image_data
"""


def MiniImageNet_CSV_Generator(dataset_dir):
    # print(dataset_dir)
    split_type = ['train', 'val', 'test']
    image_save_dir = []

    for _type in split_type:
        path = dataset_dir + '/' + _type
        image_save_dir.append(path)

    # print(image_save_dir)   # ['./mini-imagenet/train', './mini-imagenet/val', './mini-imagenet/test']
    for _path in image_save_dir:
        if os.path.exists(_path):
            shutil.rmtree(_path)
            os.mkdir(_path)
        else:
            os.mkdir(_path)

    for file in os.listdir(dataset_dir):
        pattern1 = r'[.]'
        split1 = re.split(pattern1, file)
        if split1[-1] == 'pkl':
            ori_data_path = dataset_dir + '/' + file
            ori_data = pickle.load(open(ori_data_path, 'rb'))
            labels = ori_data['class_dict']
            datas = ori_data['image_data']
            pattern2 = r'[-|.]'
            split2 = re.split(pattern2, file)

            if split2[-2] == 'train':
                csv_saver_path = dataset_dir + '/' + 'train.csv'
                image_save_path_temp = image_save_dir[0] + '/'

            if split2[-2] == 'val':
                csv_saver_path = dataset_dir + '/' + 'val.csv'
                image_save_path_temp = image_save_dir[1] + '/'

            if split2[-2] == 'test':
                csv_saver_path = dataset_dir + '/' + 'test.csv'
                image_save_path_temp = image_save_dir[2] + '/'

            csv_saver = csv.writer(open(csv_saver_path, 'w', encoding='utf-8', newline=""))
            csv_saver.writerow(['image_name', 'image_label'])
            for label in labels:
                for image_id in labels[label]:
                    jpg_name = label + '_' + str(image_id) + '.jpg'
                    csv_saver.writerow([jpg_name, label])
                    image = Image.fromarray(datas[image_id])
                    image_save_path = image_save_path_temp + jpg_name
                    image.save(image_save_path)

    return None


def Cifar_FS_CSV_Generator(dataset_dir):  # './cifar-fs/CIFAR-FS/cifar100'
    # print(dataset_dir)
    split_txt_dir = os.path.join(dataset_dir, 'splits', 'bertinetto')
    for txt in os.listdir(split_txt_dir):
        txt_path = os.path.join(split_txt_dir, txt)
        pattern = r'[.]'
        split1 = re.split(pattern, txt)
        if split1[0] == 'train':
            csv_saver_path = dataset_dir + '/' + 'train.csv'
        if split1[0] == 'val':
            csv_saver_path = dataset_dir + '/' + 'val.csv'
        if split1[0] == 'test':
            csv_saver_path = dataset_dir + '/' + 'test.csv'

        csv_saver = csv.writer(open(csv_saver_path, 'w', encoding='utf-8', newline=""))
        csv_saver.writerow(['image_name', 'image_label'])

        with open(txt_path, 'r') as f:
            for text in f.readlines():
                text = text.strip('\n')
                image_path = os.path.join(dataset_dir, 'data', text)
                for image in os.listdir(image_path):
                    image = text + '/'+image
                    csv_saver.writerow([image, text])
    return None


def Cub_200_2011_CSV_Generator(dataset_dir):
    # print(dataset_dir)
    split_txt_path = os.path.join(dataset_dir, 'train_test_split.txt')
    image_name_txt = os.path.join(dataset_dir, 'images.txt')
    image_label_txt = os.path.join(dataset_dir, 'image_class_labels.txt')
    image_names = []
    image_labels = []
    with open(image_name_txt, 'r') as nameReader:
        for img_name in nameReader.readlines():
            img_name = img_name.strip('\n').split(' ')
            image_names.append(img_name[1])
    with open(image_label_txt, 'r') as labelReader:
        for label in labelReader.readlines():
            label = label.strip('\n').split(' ')
            image_labels.append(label[1])

    csv_saver_train_path = os.path.join(dataset_dir, 'train.csv')
    csv_saver_test_path = os.path.join(dataset_dir, 'test.csv')
    csv_saver_train = csv.writer(open(csv_saver_train_path, 'w', encoding='utf-8', newline=""))
    csv_saver_train.writerow(['image_name', 'image_label'])
    csv_saver_test = csv.writer(open(csv_saver_test_path, 'w', encoding='utf-8', newline=""))
    csv_saver_test.writerow(['image_name', 'image_label'])
    with open(split_txt_path, 'r') as f:
        for text in f.readlines():
            text = text.strip('\n').split(' ')
            if text[1] == '1':
                index = int(text[0]) - 1
                csv_saver_train.writerow([image_names[index], image_labels[index]])
            else:
                index = int(text[0]) - 1
                csv_saver_test.writerow([image_names[index], image_labels[index]])

    return None


if __name__ == '__main__':
    mini_dir = './mini-imagenet'
#     cifar_dir = './cifar-fs/CIFAR-FS/cifar100'
#     cub_dir = './cub-200-2011/Caltech-UCSD Birds-200-2011/CUB_200_2011'
    MiniImageNet_CSV_Generator(mini_dir)
#     # Cifar_FS_CSV_Generator(cifar_dir)
#     # Cub_200_2011_CSV_Generator(cub_dir)
#     print('hello')
