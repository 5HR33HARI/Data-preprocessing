import streamlit as st
import glob
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import os
import shutil
import xmltodict
import json
import uuid
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import time
import threading
from multiprocessing import Process

# Set up the augmentations (just as in the original code)
rotate = iaa.Affine(rotate=(-50, 30))
rotate_180 = iaa.Affine(rotate=(180))
rotate_90 = iaa.Affine(translate_percent={"x": (0.1, -0.1), "y": (0.1, -0.1)}, rotate=(-360, 360), scale=0.5)
flip_vr = iaa.Flipud(p=1.0)
flip_hr = iaa.Fliplr(p=1.0)
move = iaa.Affine(translate_percent={"x": (0.1, -0.1), "y": (0.1, -0.1)}, scale=0.8)
contrast = iaa.GammaContrast(gamma=1.1)
resize = iaa.Resize({"height": 1280, "width": 1280})
resize1 = iaa.Resize({"height": 900, "width": 1280})
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
    "resize1": resize1
}

# Helper functions (simplified from the original script)

def read_xml2json(file):
    with open(file) as xml_file:
        my_dict = xmltodict.parse(xml_file.read())
    return my_dict

def read_image(jason, source_path):
    img_file = jason['annotation']['filename']
    image = imageio.imread(source_path + '/' + img_file)
    return image

def write_image(dest_path, image_aug, img_name):
    imageio.imwrite(dest_path + img_name, image_aug)

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

    if isinstance(jason['annotation']['object'], list):
        for i, z in enumerate(d['annotation']["object"]):
            if ((z['bndbox']['xmax']) <= shape[1]) & ((z['bndbox']['ymax']) <= shape[0]):
                if ((z['bndbox']['xmax']) >= 0) & ((z['bndbox']['ymax']) >= 0) & ((z['bndbox']['xmin']) >= 0) & ((z['bndbox']['ymin']) >= 0):
                    exec(f'object_{i} = e.SubElement(r, "object")')
                    exec(f'e.SubElement(object_{i}, "name").text = str(z["name"])')
                    exec(f'e.SubElement(object_{i}, "pose").text = str(z["pose"])')
                    exec(f'e.SubElement(object_{i}, "truncated").text = str(z["truncated"])')
                    exec(f'e.SubElement(object_{i}, "difficult").text = str(z["difficult"])')
                    exec(f'bndbox_{i} = e.SubElement(object_{i}, "bndbox")')
                    exec(f"e.SubElement(bndbox_{i}, 'xmin').text = str(z['bndbox']['xmin'])")
                    exec(f"e.SubElement(bndbox_{i}, 'ymin').text = str(z['bndbox']['ymin'])")
                    exec(f"e.SubElement(bndbox_{i}, 'xmax').text = str(z['bndbox']['xmax'])")
                    exec(f"e.SubElement(bndbox_{i}, 'ymax').text = str(z['bndbox']['ymax'])")
    a = e.ElementTree(r)
    a.write(dest_path + xml_name)

def aug_img_bndbox(source_path, dest_path, augment_list):
    del_create_directory(dest_path)
    for xml in sorted(glob.glob(source_path + '/*.xml')):
        for aug in augment_list:
            jason = read_xml2json(xml)
            image = read_image(jason, source_path)
            augment = augmentations[aug]
            object_list, image_aug, shape = objects_coord_aug(jason, augment, image)
            jason, img_name, xml_name = edit_jason(jason, object_list, aug, shape)
            write_xml(jason, xml_name, dest_path, shape)
            write_image(dest_path, image_aug, img_name)

# Streamlit UI
def main():
    st.title("Image Augmentation for Object Detection")

    st.sidebar.header("Configuration")

    # Uploading the folder containing XML and image files
    uploaded_dir = st.sidebar.file_uploader("Upload a ZIP file containing images and XML annotations", type="zip", key="upload")
    if uploaded_dir:
        with open("uploaded.zip", "wb") as f:
            f.write(uploaded_dir.getbuffer())
        shutil.unpack_archive("uploaded.zip", "uploaded_data")
        st.success("Files uploaded and extracted successfully!")

    # Selecting augmentations to apply
    augment_list = st.sidebar.multiselect("Choose augmentations to apply", list(augmentations.keys()), default=["rotate", "flip_hr", "resize"])

    if st.button("Start Augmentation"):
        if uploaded_dir and augment_list:
            start_time = time.time()
            source_path = "uploaded_data"  # Adjust this to your extracted directory
            dest_path = "augmented_data"  # Folder to save augmented images and XMLs
            aug_img_bndbox(source_path, dest_path, augment_list)
            end_time = time.time()
            st.success(f"Augmentation complete! Time taken: {end_time - start_time:.2f} seconds")

            st.write("Augmented images are saved in the folder: `augmented_data`")

            # Show a sample augmented image
            sample_image = glob.glob(f"{dest_path}/*.jpg")[0]  # Assuming images are saved as .jpg
            st.image(sample_image, caption="Sample Augmented Image", use_column_width=True)

if __name__ == "__main__":
    main()
