import argparse
import os
import glob
from typing import Mapping
import cv2
import shutil
from datetime import datetime
from xml.dom import minidom
from transformer import Transformer
import sys
import platform
import xml.etree.ElementTree as ET

cwd = os.getcwd()

project_name ="ABCD_mod"
data_folder = cwd+'/annotations/'
# data_folder = 



if not data_folder.endswith('/'):
	data_folder += '/'

## Create data sr=tructure
def create_data_structure(root_folder):

	if not os.path.isdir(root_folder):

		os.makedirs(root_folder+'/images')
		os.makedirs(root_folder+'/images/test') 
		os.makedirs(root_folder+'/images/train')

		os.makedirs(root_folder+'/labels')
		os.makedirs(root_folder+'/labels/test')
		os.makedirs(root_folder+'/labels/train')
	else:
		print(root_folder+' already exists.... ')


	return root_folder+'/images/test', root_folder+'/images/train'

test_path, train_path =  create_data_structure(project_name)


print(test_path,train_path,'*************************')



## Create yaml file

def create_yaml_file(file):
	classes = []
	fr = open(file)
	for line in fr:
		line = line.strip()
		print(line)
		if not line == '':
			classes.append(line)

	fr.close()
	fw = open(project_name+'.yaml','w')
	test_path, train_path = create_data_structure(project_name)
	fw.write('train: '+'./'+train_path+'\n') # train: images/train2017  # train images (relative to 'path') 128 images
	fw.write('val: '+'./'+test_path+'\n') # images/train2017  # val images (relative to 'path') 128 images

	# Classes
	fw.write('nc: '+str(len(classes))+'\n') # number of classes
	fw.write('names: '+str(classes))

	fw.close()
	return classes
# classes = create_yaml_file('classes.txt')


## test and train seperation 
def test_train_split(path):

	res = os.listdir(path)

	length = (len(res)/2 ) / 5

	count = 0
	test_path, train_path = create_data_structure(project_name)

	for file in res:
		if file.endswith('.jpg') or file.endswith('.png'):
			count += 1
			if count <= length:
				# print('copying into test..........',count)
				shutil.copyfile(path+file,test_path+'/'+file)
				shutil.copyfile(path+file.split('.')[0]+'.xml',test_path+'/'+file.split('.')[0]+'.xml')

			else:
				# print('copying into train.......',count)
				shutil.copyfile(path+file,train_path+'/'+file)
				shutil.copyfile(path+file.split('.')[0]+'.xml',train_path+'/'+file.split('.')[0]+'.xml')


def test_train(data_folder):
	res = os.walk(data_folder)
	print(res)

	for i in res:
		root = i[0]
		folders = i[1]
		for folder in folders:
			print(root+folder+'/')
			test_train_split(root+folder+'/')

test_train(data_folder)




## Create classes.txt file
def all_class_names(path):
	class_names = []
	files = os.listdir(path)
	for file in files:
		if file.endswith('.xml'):
			tree = ET.parse(path+file)
			root = tree.getroot()
			for elt in root.iter():
				if elt.tag == 'name':
					class_names.append(elt.text)
					# print(file)

	return class_names


def create_txt_file():
	
	all_classes = []
	for c in [train_path,test_path]:
		classes = all_class_names('./'+c+'/')
		all_classes.extend(classes)		
	
	fw = open('classes.txt','w')
	for i in set(all_classes):
		fw.write(i+'\n')
	fw.close()

create_txt_file()
classes = create_yaml_file('classes.txt')



def xml2txt(xml_dir,out_dir):
    parser = argparse.ArgumentParser(description="Formatter from ImageNet xml to Darknet text format")
    parser.add_argument("-xml", help="Relative location of xml files directory" ,default='xml')
    parser.add_argument("-out", help="Relative location of output txt files directory", default="out")
    parser.add_argument("-c", help="Relative path to classes file", default="classes.txt")
    args = parser.parse_args()

    xml_dir = os.path.join(os.path.dirname(os.path.realpath('__file__')), xml_dir)
    if not os.path.exists(xml_dir):
        print("Provide the correct folder for xml files.")
        sys.exit()

    out_dir = os.path.join(os.path.dirname(os.path.realpath('__file__')), out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.access(out_dir, os.W_OK):
        print("%s folder is not writeable." % out_dir)
        sys.exit()
    
    
    transformer = Transformer(xml_dir=xml_dir, out_dir=out_dir)
    transformer.transform()

for i in ['./'+test_path,'./'+train_path]:
	xml2txt(i,i.replace('images','labels'))

print('successfully converted xml to txt ')
print('created ',project_name+'.yaml file')

