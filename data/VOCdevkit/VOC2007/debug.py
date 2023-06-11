import argparse
import sys
import cv2
import os

import os.path          as osp
import numpy            as np

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree  as ET


parser    = argparse.ArgumentParser(
            description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()

parser.add_argument('--root', default='E:\PythonCode\ssd.pytorch-master\data\VOCdevkit\VOC2007', help='Dataset root directory path')

args = parser.parse_args()

CLASSES = (  # always index 0
    # 'aeroplane', 'bicycle', 'bird', 'boat',
    # 'bottle', 'bus', 'car', 'cat', 'chair',
    # 'cow', 'diningtable', 'dog', 'horse',
    # 'motorbike', 'person', 'pottedplant',
    # 'sheep', 'sofa', 'train', 'tvmonitor',
    'face', 'face_mask')

annopath = osp.join('%s', 'Annotations', '%s.{}'.format("xml"))
imgpath  = osp.join('%s', 'JPEGImages',  '%s.{}'.format("jpg"))

def vocChecker(image_id, width, height, keep_difficult = False):
    target   = ET.parse(annopath % image_id).getroot()
    res      = []

    for obj in target.iter('object'):

        difficult = int(obj.find('difficult').text) == 1

        if not keep_difficult and difficult:
            continue

        name = obj.find('name').text.lower().strip()
        bbox = obj.find('bndbox')

        pts    = ['xmin', 'ymin', 'xmax', 'ymax']
        bndbox = []

        for i, pt in enumerate(pts):

            cur_pt = int(bbox.find(pt).text) - 1
            # scale height or width
            cur_pt = float(cur_pt) / width if i % 2 == 0 else float(cur_pt) / height

            bndbox.append(cur_pt)

        print(name)
        label_idx =  dict(zip(CLASSES, range(len(CLASSES))))[name]
        bndbox.append(label_idx)
        res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
        # img_id = target.find('filename').text[:-4]
    print(res)
    try :
        print(np.array(res)[:,4])
        print(np.array(res)[:,:4])
    except IndexError:
        print("\nINDEX ERROR HERE !\n")
        exit(0)
    return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

if __name__ == '__main__' :

    i = 0

    for name in sorted(os.listdir(osp.join(args.root,'Annotations'))):
    # as we have only one annotations file per image
        i += 1

        img    = cv2.imread(imgpath  % (args.root,name.split('.')[0]))
        height, width, channels = img.shape
        print("path : {}".format(annopath % (args.root,name.split('.')[0])))
        res = vocChecker((args.root, name.split('.')[0]), height, width)
    print("Total of annotations : {}".format(i))