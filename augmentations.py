# augmentations.py
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import os
import shutil
import xmltodict
import json
import uuid
import xml.etree.cElementTree as e
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import glob

# Augmentation definitions
rotate = iaa.Affine(rotate=(-50, 30))
rotate_180 = iaa.Affine(rotate=(180))
rotate_90 = iaa.Affine(translate_percent={"x": (0.1, -0.1), "y": (0.1, -0.1)}, rotate=(-360, 360), scale=0.5)
flip_vr = iaa.Flipud(p=1.0)
flip_hr = iaa.Fliplr(p=1.0)
move = iaa.Affine(translate_percent={"x": (0.1, -0.1), "y": (0.1, -0.1)}, scale=0.8)
contrast = iaa.GammaContrast(gamma=1.1)
resize = iaa.Resize({"height": 1280, "width": 1280})
crop = iaa.CenterCropToFixedSize(height=200, width=460)
brightness = iaa.WithBrightnessChannels(iaa.Add((20, 40)))
drop = iaa.Dropout2d(p=0.5)
blend = iaa.BlendAlpha(0.1, iaa.Affine(translate_percent={"x": (0.1, -0.1), "y": (0.1, -0.1)}, rotate=(-360, 360), scale=0.5), per_channel=0.5)
noise = iaa.ImpulseNoise(0.1)

augmentations = {
    "blend": blend,
    "rotate_90": rotate_90,
    "rotate_180": rotate_180,
    'rotate': rotate,
    'flip_hr': flip_hr,
    'flip_vr': flip_vr,
    'affine': move,
    'contrast': contrast,
    'resize': resize,
    'crop': crop,
    'brightness': brightness,
    "drop": drop,
    "noise": noise,
}

def create_directory(directory):
    try:
        os.makedirs(directory)
    except:
        print('Directory already exists!')

def del_create_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
        print(directory, 'removed!')
    os.makedirs(directory)
    print(directory, 'created!')

def read_xml2json(file):
    with open(file) as xml_file:
        my_dict = xmltodict.parse(xml_file.read())
    return json.loads(json.dumps(my_dict))

def read_image(jason, source_path):
    img_file = jason['annotation']['filename']
    return imageio.imread(source_path + '/' + img_file)

def unique_id():
    return uuid.uuid4()

def aug_rename_file(jason, aug):
    id = str(aug)
    img_file = jason['annotation']['filename']
    img_name = img_file.replace(img_file[-4:], f"_{id}{img_file[-4:]}")
    xml_name = img_name.replace(img_name[-4:], '.xml')
    return img_name, xml_name

def multi_objects_coord_aug(jason, augment, image):
    bbs_list = []
    for obj in jason['annotation']['object']:
        x1, y1, x2, y2 = int(obj['bndbox']['xmin']), int(obj['bndbox']['ymin']), int(obj['bndbox']['xmax']), int(obj['bndbox']['ymax'])
        bbs_list.append(BoundingBox(x1=x1, x2=x2, y1=y1, y2=y2, label=str(obj['name'])))
    bbs = BoundingBoxesOnImage(bbs_list, shape=image.shape)
    image_aug, bbs_aug = augment(image=image, bounding_boxes=(bbs))
    object_list = []
    for i, obj in enumerate(jason['annotation']['object']):
        xmin, ymin, xmax, ymax = bbs_aug[i].x1_int, bbs_aug[i].y1_int, bbs_aug[i].x2_int, bbs_aug[i].y2_int
        object_dict = {'name': obj['name'], 'pose': 'Unspecified', 'truncated': '0', 'difficult': '0', 'bndbox': {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}}
        object_list.append(object_dict)
    return object_list, image_aug

def write_xml(jason, xml_name, dest_path, shape):
    d = jason
    r = e.Element("annotation")
    e.SubElement(r, "folder").text = d['annotation']["folder"]
    e.SubElement(r, "filename").text = d['annotation']["filename"]
    e.SubElement(r, "path").text = str(d['annotation']["path"])
    source_ = e.SubElement(r, "source")
    e.SubElement(source_, 'database').text = str(d['annotation']["source"]['database'])
    size_ = e.SubElement(r, "size")
    e.SubElement(size_, 'width').text = str(d['annotation']["size"]['width'])
    e.SubElement(size_, 'height').text = str(d['annotation']["size"]['height'])
    e.SubElement(size_, 'depth').text = str(d['annotation']["size"]['depth'])
    e.SubElement(r, "segmented").text = str(d['annotation']["segmented"])

    for i, obj in enumerate(d['annotation']["object"]):
        if ((obj['bndbox']['xmax'] <= shape[1]) & (obj['bndbox']['ymax'] <= shape[0])):
            object_ = e.SubElement(r, "object")
            e.SubElement(object_, "name").text = str(obj["name"])
            e.SubElement(object_, "pose").text = str(obj["pose"])
            e.SubElement(object_, "truncated").text = str(obj["truncated"])
            e.SubElement(object_, "difficult").text = str(obj["difficult"])
            bndbox_ = e.SubElement(object_, "bndbox")
            e.SubElement(bndbox_, 'xmin').text = str(obj['bndbox']['xmin'])
            e.SubElement(bndbox_, 'ymin').text = str(obj['bndbox']['ymin'])
            e.SubElement(bndbox_, 'xmax').text = str(obj['bndbox']['xmax'])
            e.SubElement(bndbox_, 'ymax').text = str(obj['bndbox']['ymax'])
    
    a = e.ElementTree(r)
    a.write(dest_path + xml_name)

def write_image(dest_path, image_aug, img_name):
    imageio.imwrite(dest_path + img_name, image_aug)

def aug_img_bndbox(source_path, dest_path, augment_list):
    del_create_directory(dest_path)
    for xml_file in sorted(glob.glob(source_path + '/*.xml')):
        jason = read_xml2json(xml_file)
        image = read_image(jason, source_path)
        for aug in augment_list:
            augment = augmentations[aug]
            object_list, image_aug = multi_objects_coord_aug(jason, augment, image)
            jason, img_name, xml_name = aug_rename_file(jason, aug)
            write_xml(jason, xml_name, dest_path, image.shape)
            write_image(dest_path, image_aug, img_name)
            print(f"{xml_file} augmented with {aug}")

