import os
import json
import numpy as np
from os import listdir
from os.path import isfile, join
from collections import OrderedDict
from matplotlib import pyplot as plt
# /mnt/bess/shared/tale2/Shared/schobs/data/ISBI2015_landmarks/images/366.nii.gz
import csv
from PIL import Image
import nibabel as nib
import random


def transpose_all_nibs(nib_path):
    all_ims = [f for f in listdir(nib_path) if isfile(join(nib_path, f)) and "nii.gz" in join(nib_path, f) ]

    for im in all_ims:
        print(im)
        original_nifti =  nib.load(os.path.join(nib_path, im))
        # print("og nifit ", original_nifti)
        transposed_array= np.transpose(nib.load(os.path.join(nib_path, im)).get_fdata())

        new_nifti = nib.Nifti1Image(transposed_array, np.eye(4))

        # print("new nifti data ", new_nifti)
        # plt.imshow(new_nifti.get_fdata())
        # plt.show()
        # plt.show()
        nib.save(new_nifti, os.path.join(nib_path, im))

def ISBI2015_to_json(path_to_jun_annotations, path_to_fold_infos, path_to_images, output_path, root_path, num_folds=4):
    # anno =  list(csv.reader(open(path_to_jun_annotations, "r").read(), delimiter=";"))
    anno =np.loadtxt(open(path_to_jun_annotations, "rb"), delimiter=",")
    
    for fold in range(num_folds):
        this_train_idx_path = os.path.join(path_to_fold_infos ,"set"+str(fold+1),'train.txt') #bc 0 indexing
        this_valid_idx_path = os.path.join(path_to_fold_infos , "set"+str(fold+1),'val.txt') #bc 0 indexing

        # xs = open(this_train_idx_path, "r").read().splitlines()
        # for i, c in enumerate(xs):
        #     print(i, c)
        # # print(x.strip())
        # train_idx = [int(x) for x in xs]
        # # valid_idx = [int(x) for x in open(this_valid_idx_path, "r").read()]
        train_idx = open(this_train_idx_path, "r").read().splitlines()
        valid_idx = open(this_valid_idx_path, "r").read().splitlines()
        
        data = OrderedDict()

        #basic info
        data['name'] = 'ISBI2015 JUNIOR'
        data['desciption'] = 'Cephalograms with 19 annotated landmarks from the JUNIOR annotators from the ISBI 2015 Cephalometric X-Ray Image Analysis Challenge.'
        data['fold'] = fold

        data['training'] = []

        for idx in train_idx:
            # print(idx)
            inner_dict = {}
            inner_dict['id'] = idx
            

            img_link = os.path.join(path_to_images, idx+ '.nii.gz')

            # plt.imshow(im_array)
            # plt.show()

            # plt.imshow(im_array)
            # plt.show()

            #need to rotate these images to match the landmarks:
            this_im_coords = anno[int(idx)-1]
            coordinates = []
            count = 1 # start at 1 instead of 0 to ignore the first column indicating the index
            for c in range(int(len(this_im_coords)/2)):
                coordinates.append([this_im_coords[count]*10, this_im_coords[(count)+1]*10]) #multiply by 10bevause they are downsacled by 10 for some reason in this file.
                count+=2
            inner_dict['coordinates'] = coordinates
            inner_dict['image'] = img_link

            data['training'].append(inner_dict)


        data['testing'] = []
        for idx in valid_idx:
            # print(idx)
            inner_dict = {}
            inner_dict['id'] = idx
            

            img_link = os.path.join(path_to_images, idx+ '.nii.gz')
            this_im_coords = anno[int(idx)-1]
            coordinates = []
            count = 1 # start at 1 instead of 0 to ignore the first column indicating the index
            for c in range(int(len(this_im_coords)/2)):

                coordinates.append([this_im_coords[count]*10, this_im_coords[(count)+1]*10]) #multiply by 10bevause they are downsacled by 10 for some reason in this file.
                count+=2
            inner_dict['coordinates'] = coordinates
            inner_dict['image'] = img_link

            data['testing'].append(inner_dict)

       
                # print(train_idx)
        with open(os.path.join(output_path, 'fold'+str(fold)+'.json'), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)




def ISBI2015_to_json_with_vals(path_to_jun_annotations, path_to_fold_infos, path_to_images, output_path, root_path, num_folds=4):

    '''
    generate json from annotations of ISBI 2015 cephalometric dataset. 
    We randomly pick 20% of the training set for validation for early stopping.
    
    '''
    # anno =  list(csv.reader(open(path_to_jun_annotations, "r").read(), delimiter=";"))
    os.makedirs((output_path +'/w_valid/'), exist_ok=True)  
    anno =np.loadtxt(open(path_to_jun_annotations, "rb"), delimiter=",")
    
    for fold in range(num_folds):
        this_train_idx_path = os.path.join(path_to_fold_infos ,"set"+str(fold+1),'train.txt') #bc 0 indexing
        this_test_idx_path = os.path.join(path_to_fold_infos , "set"+str(fold+1),'val.txt') #bc 0 indexing

        
        train_idx = open(this_train_idx_path, "r").read().splitlines()
        test_idx = open(this_test_idx_path, "r").read().splitlines()
        
        data = OrderedDict()

        #basic info
        data['name'] = 'ISBI2015 JUNIOR'
        data['desciption'] = 'Cephalograms with 19 annotated landmarks from the JUNIOR annotators from the ISBI 2015 Cephalometric X-Ray Image Analysis Challenge.'
        data['fold'] = fold

        data['training'] = []
        data['validation'] = []

        
        valid_idx = random.sample(train_idx, int(len(train_idx)/5))

        print("val len and idxs: ", len(valid_idx), valid_idx, "\n")
        for idx in train_idx:
            # print(idx)
            inner_dict = {}
            inner_dict['id'] = idx
            

            img_link = os.path.join(path_to_images, idx+ '.nii.gz')

            # plt.imshow(im_array)
            # plt.show()

            # plt.imshow(im_array)
            # plt.show()

            #need to rotate these images to match the landmarks:
            this_im_coords = anno[int(idx)-1]
            coordinates = []
            count = 1 # start at 1 instead of 0 to ignore the first column indicating the index
            for c in range(int(len(this_im_coords)/2)):
                coordinates.append([this_im_coords[count]*10, this_im_coords[(count)+1]*10]) #multiply by 10bevause they are downsacled by 10 for some reason in this file.
                count+=2
            inner_dict['coordinates'] = coordinates
            inner_dict['image'] = img_link

            if idx in valid_idx:
                data['validation'].append(inner_dict)
            else:
                data['training'].append(inner_dict)


        data['testing'] = []
        for idx in test_idx:
            # print(idx)
            inner_dict = {}
            inner_dict['id'] = idx
            

            img_link = os.path.join(path_to_images, idx+ '.nii.gz')
            this_im_coords = anno[int(idx)-1]
            coordinates = []
            count = 1 # start at 1 instead of 0 to ignore the first column indicating the index
            for c in range(int(len(this_im_coords)/2)):

                coordinates.append([this_im_coords[count]*10, this_im_coords[(count)+1]*10]) #multiply by 10bevause they are downsacled by 10 for some reason in this file.
                count+=2
            inner_dict['coordinates'] = coordinates
            inner_dict['image'] = img_link

            data['testing'].append(inner_dict)

       
                # print(train_idx)
        with open(output_path+ '/w_valid/fold'+str(fold)+'.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


path_to_jun_annotations= "/mnt/tale_shared/schobs/data/ISBI2015_landmarks/setup_ann/all_landmarks/all_junior.csv"
path_to_fold_infos = "/mnt/tale_shared/schobs/data/ISBI2015_landmarks/setup_ann/all_landmarks/cv/"
path_to_images =  "images/"
output_path = "/mnt/tale_shared/schobs/data/ISBI2015_landmarks/lann_folds/"
root_path="/mnt/bess/shared/tale2/Shared/schobs/data/ISBI2015_landmarks" 



# transpose_all_nibs(os.path.join(root_path, path_to_images))
# ISBI2015_to_json(path_to_jun_annotations, path_to_fold_infos, path_to_images, output_path, root_path)
ISBI2015_to_json_with_vals(path_to_jun_annotations, path_to_fold_infos, path_to_images, output_path, root_path)
