import os
import json
from re import U
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
import pandas as pd
from pyrsistent import v
import pydicom as dicom
import matplotlib.patches as patches
import glob


def transpose_all_nibs(nib_path):
    all_ims = [
        f
        for f in listdir(nib_path)
        if isfile(join(nib_path, f)) and "nii.gz" in join(nib_path, f)
    ]

    for im in all_ims:
        print(im)
        original_nifti = nib.load(os.path.join(nib_path, im))
        # print("og nifit ", original_nifti)
        transposed_array = np.transpose(
            nib.load(os.path.join(nib_path, im)).get_fdata()
        )

        new_nifti = nib.Nifti1Image(transposed_array, np.eye(4))

        # print("new nifti data ", new_nifti)
        # plt.imshow(new_nifti.get_fdata())
        # plt.show()
        # plt.show()
        nib.save(new_nifti, os.path.join(nib_path, im))


def ISBI2015_to_json(
    path_to_jun_annotations,
    path_to_fold_infos,
    path_to_images,
    output_path,
    root_path,
    num_folds=4,
):
    # anno =  list(csv.reader(open(path_to_jun_annotations, "r").read(), delimiter=";"))
    anno = np.loadtxt(open(path_to_jun_annotations, "rb"), delimiter=",")

    for fold in range(num_folds):
        this_train_idx_path = os.path.join(
            path_to_fold_infos, "set" + str(fold + 1), "train.txt"
        )  # bc 0 indexing
        this_valid_idx_path = os.path.join(
            path_to_fold_infos, "set" + str(fold + 1), "val.txt"
        )  # bc 0 indexing

        # xs = open(this_train_idx_path, "r").read().splitlines()
        # for i, c in enumerate(xs):
        #     print(i, c)
        # # print(x.strip())
        # train_idx = [int(x) for x in xs]
        # # valid_idx = [int(x) for x in open(this_valid_idx_path, "r").read()]
        train_idx = open(this_train_idx_path, "r").read().splitlines()
        valid_idx = open(this_valid_idx_path, "r").read().splitlines()

        data = OrderedDict()

        # basic info
        data["name"] = "ISBI2015 JUNIOR"
        data[
            "desciption"
        ] = "Cephalograms with 19 annotated landmarks from the JUNIOR annotators from the ISBI 2015 Cephalometric X-Ray Image Analysis Challenge."
        data["fold"] = fold

        data["training"] = []

        for idx in train_idx:
            # print(idx)
            inner_dict = {}
            inner_dict["id"] = idx

            img_link = os.path.join(path_to_images, idx + ".nii.gz")

            # plt.imshow(im_array)
            # plt.show()

            # plt.imshow(im_array)
            # plt.show()

            # need to rotate these images to match the landmarks:
            this_im_coords = anno[int(idx) - 1]
            coordinates = []
            count = 1  # start at 1 instead of 0 to ignore the first column indicating the index
            for c in range(int(len(this_im_coords) / 2)):
                coordinates.append(
                    [this_im_coords[count] * 10, this_im_coords[(count) + 1] * 10]
                )  # multiply by 10bevause they are downsacled by 10 for some reason in this file.
                count += 2
            inner_dict["coordinates"] = coordinates
            inner_dict["image"] = img_link

            data["training"].append(inner_dict)

        data["testing"] = []
        for idx in valid_idx:
            # print(idx)
            inner_dict = {}
            inner_dict["id"] = idx

            img_link = os.path.join(path_to_images, idx + ".nii.gz")
            this_im_coords = anno[int(idx) - 1]
            coordinates = []
            count = 1  # start at 1 instead of 0 to ignore the first column indicating the index
            for c in range(int(len(this_im_coords) / 2)):

                coordinates.append(
                    [this_im_coords[count] * 10, this_im_coords[(count) + 1] * 10]
                )  # multiply by 10bevause they are downsacled by 10 for some reason in this file.
                count += 2
            inner_dict["coordinates"] = coordinates
            inner_dict["image"] = img_link

            data["testing"].append(inner_dict)

            # print(train_idx)
        with open(
            os.path.join(output_path, "fold" + str(fold) + ".json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


def ISBI2015_to_json_with_vals(
    path_to_jun_annotations,
    path_to_fold_infos,
    path_to_images,
    output_path,
    root_path,
    val_size=0.2,
    num_folds=4,
):
    """
    generate json from annotations of ISBI 2015 cephalometric dataset.
    We randomly pick 20% of the training set for validation for early stopping.

    """
    # anno =  list(csv.reader(open(path_to_jun_annotations, "r").read(), delimiter=";"))
    os.makedirs((output_path + "/w_valid/"), exist_ok=True)
    anno = np.loadtxt(open(path_to_jun_annotations, "rb"), delimiter=",")

    for fold in range(num_folds):
        this_train_idx_path = os.path.join(
            path_to_fold_infos, "set" + str(fold + 1), "train.txt"
        )  # bc 0 indexing
        this_test_idx_path = os.path.join(
            path_to_fold_infos, "set" + str(fold + 1), "val.txt"
        )  # bc 0 indexing

        train_idx = open(this_train_idx_path, "r").read().splitlines()
        test_idx = open(this_test_idx_path, "r").read().splitlines()

        data = OrderedDict()

        # basic info
        data["name"] = "ISBI2015 JUNIOR"
        data[
            "desciption"
        ] = "Cephalograms with 19 annotated landmarks from the JUNIOR annotators from the ISBI 2015 Cephalometric X-Ray Image Analysis Challenge."
        data["fold"] = fold

        data["training"] = []
        data["validation"] = []

        valid_idx = random.sample(train_idx, int(len(train_idx) * val_size))

        print("val len and idxs: ", len(valid_idx), valid_idx, "\n")
        for idx in train_idx:
            # print(idx)
            inner_dict = {}
            inner_dict["id"] = idx

            img_link = os.path.join(path_to_images, idx + ".nii.gz")

            # plt.imshow(im_array)
            # plt.show()

            # plt.imshow(im_array)
            # plt.show()

            # need to rotate these images to match the landmarks:
            this_im_coords = anno[int(idx) - 1]
            coordinates = []
            count = 1  # start at 1 instead of 0 to ignore the first column indicating the index
            for c in range(int(len(this_im_coords) / 2)):
                coordinates.append(
                    [this_im_coords[count] * 10, this_im_coords[(count) + 1] * 10]
                )  # multiply by 10bevause they are downsacled by 10 for some reason in this file.
                count += 2
            inner_dict["coordinates"] = coordinates
            inner_dict["image"] = img_link

            if idx in valid_idx:
                data["validation"].append(inner_dict)
            else:
                data["training"].append(inner_dict)

        data["testing"] = []
        for idx in test_idx:
            # print(idx)
            inner_dict = {}
            inner_dict["id"] = idx

            img_link = os.path.join(path_to_images, idx + ".nii.gz")
            this_im_coords = anno[int(idx) - 1]
            coordinates = []
            count = 1  # start at 1 instead of 0 to ignore the first column indicating the index
            for c in range(int(len(this_im_coords) / 2)):

                coordinates.append(
                    [this_im_coords[count] * 10, this_im_coords[(count) + 1] * 10]
                )  # multiply by 10bevause they are downsacled by 10 for some reason in this file.
                count += 2
            inner_dict["coordinates"] = coordinates
            inner_dict["image"] = img_link

            data["testing"].append(inner_dict)

            # print(train_idx)
        with open(
            output_path + "/w_valid/fold" + str(fold) + ".json", "w", encoding="utf-8"
        ) as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


def ISBI2015_to_json_with_vals_vary_train(
    path_to_jun_annotations,
    path_to_fold_infos,
    path_to_images,
    output_path,
    val_size=0.1,
    num_folds=4,
):
    """
    generate json from annotations of ISBI 2015 cephalometric dataset.
    We randomly pick 10% of the training set for validation for early stopping.

    """
    # anno =  list(csv.reader(open(path_to_jun_annotations, "r").read(), delimiter=";"))
    os.makedirs((output_path + "/w_valid/"), exist_ok=True)
    anno = np.loadtxt(open(path_to_jun_annotations, "rb"), delimiter=",")

    for fold in range(num_folds):
        print("fold: ", fold)

        this_train_idx_path = os.path.join(
            path_to_fold_infos, "set" + str(fold + 1), "train.txt"
        )
        this_train_idx_path = os.path.join(
            path_to_fold_infos, "set" + str(fold + 1), "train.txt"
        )  # bc 0 indexing
        this_test_idx_path = os.path.join(
            path_to_fold_infos, "set" + str(fold + 1), "val.txt"
        )  # bc 0 indexing

        train_idx = open(this_train_idx_path, "r").read().splitlines()
        test_idx = open(this_test_idx_path, "r").read().splitlines()

        valid_idx = random.sample(train_idx, int(len(train_idx) * val_size))

        only_train_idx = [x for x in train_idx if x not in valid_idx]
        print("val len and only_train len: ", len(valid_idx), len(only_train_idx), "\n")

        for train_set_size in [0.25, 0.5, 0.75, 1.0]:
            print("\n \n For train set size: ", train_set_size)
            # size_of_training = int(len(train_idx)*train_set_size)
            shuffled_idx = random.sample(
                only_train_idx, int(len(only_train_idx) * train_set_size)
            )
            print("train idx len ", len(shuffled_idx))

            data = OrderedDict()

            # basic info
            data["name"] = "ISBI2015 JUNIOR"
            data[
                "desciption"
            ] = "Cephalograms with 19 annotated landmarks from the JUNIOR annotators from the ISBI 2015 Cephalometric X-Ray Image Analysis Challenge."
            data["fold"] = fold

            data["training"] = []
            data["validation"] = []

            for idx in train_idx:
                # print(idx)
                inner_dict = {}
                inner_dict["id"] = idx

                img_link = os.path.join(path_to_images, idx + ".nii.gz")

                # plt.imshow(im_array)
                # plt.show()

                # plt.imshow(im_array)
                # plt.show()

                # need to rotate these images to match the landmarks:
                this_im_coords = anno[int(idx) - 1]
                coordinates = []
                count = 1  # start at 1 instead of 0 to ignore the first column indicating the index
                for c in range(int(len(this_im_coords) / 2)):
                    coordinates.append(
                        [this_im_coords[count] * 10, this_im_coords[(count) + 1] * 10]
                    )  # multiply by 10bevause they are downsacled by 10 for some reason in this file.
                    count += 2
                inner_dict["coordinates"] = coordinates
                inner_dict["image"] = img_link

                if idx in valid_idx:
                    data["validation"].append(inner_dict)
                elif idx in shuffled_idx:
                    data["training"].append(inner_dict)

            data["testing"] = []
            for idx in test_idx:
                # print(idx)
                inner_dict = {}
                inner_dict["id"] = idx

                img_link = os.path.join(path_to_images, idx + ".nii.gz")
                this_im_coords = anno[int(idx) - 1]
                coordinates = []
                count = 1  # start at 1 instead of 0 to ignore the first column indicating the index
                for c in range(int(len(this_im_coords) / 2)):

                    coordinates.append(
                        [this_im_coords[count] * 10, this_im_coords[(count) + 1] * 10]
                    )  # multiply by 10bevause they are downsacled by 10 for some reason in this file.
                    count += 2
                inner_dict["coordinates"] = coordinates
                inner_dict["image"] = img_link

                data["testing"].append(inner_dict)

            print(
                "lens train, val, test: ",
                len(data["training"]),
                len(data["validation"]),
                len(data["testing"]),
            )
            # print(train_idx)
            os.makedirs(
                output_path + "/w_valid10_trainsize" + str(train_set_size),
                exist_ok=True,
            )

            print(
                "saving to",
                output_path
                + "/w_valid10_trainsize"
                + str(train_set_size)
                + "/fold"
                + str(fold)
                + ".json",
            )
            with open(
                output_path
                + "/w_valid10_trainsize"
                + str(train_set_size)
                + "/fold"
                + str(fold)
                + ".json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(data, f, ensure_ascii=False, indent=4)


def ISBI2015_to_json_with_vals_challenge(
    path_to_jun_annotations,
    path_to_sen_annotations,
    path_to_train_idx,
    path_to_test1_idx,
    path_to_test2_idx,
    path_to_images,
    output_path,
    val_size=0.1,
):
    """
    generate json from annotations of ISBI 2015 cephalometric dataset.
    We randomly pick 20% of the training set for validation for early stopping.

    """
    # anno =  list(csv.reader(open(path_to_jun_annotations, "r").read(), delimiter=";"))
    os.makedirs((output_path + "/w_valid/"), exist_ok=True)
    anno_jun = np.loadtxt(open(path_to_jun_annotations, "rb"), delimiter=",")
    anno_sen = np.loadtxt(open(path_to_sen_annotations, "rb"), delimiter=",")

    train_idx = open(path_to_train_idx, "r").read().splitlines()
    test1_idx = open(path_to_test1_idx, "r").read().splitlines()
    test2_idx = open(path_to_test2_idx, "r").read().splitlines()

    data = OrderedDict()

    # basic info
    data["name"] = "ISBI2015 CHALLENGE"
    data[
        "desciption"
    ] = "Cephalograms with 19 annotated landmarks from the (JUNIOR+SENIOR)/2 annotators from the ISBI 2015 Cephalometric X-Ray Image Analysis Challenge."
    data["fold"] = 1

    data["training"] = []
    data["validation"] = []
    # data['testing'] = []

    valid_idx = random.sample(train_idx, int(len(train_idx) * val_size))

    print("val len and idxs: ", len(valid_idx), valid_idx, "\n")
    for idx in train_idx:
        # print(idx)
        inner_dict = {}
        inner_dict["id"] = idx

        img_link = os.path.join(path_to_images, idx + ".nii.gz")

        # need to rotate these images to match the landmarks:
        this_im_coords_jun = anno_jun[int(idx) - 1]
        this_im_coords_sen = anno_sen[int(idx) - 1]

        coordinates = []
        count = (
            1  # start at 1 instead of 0 to ignore the first column indicating the index
        )
        for c in range(int(len(this_im_coords_jun) / 2)):
            jun_ann = [
                this_im_coords_jun[count] * 10,
                this_im_coords_sen[(count) + 1] * 10,
            ]
            sen_ann = [
                this_im_coords_sen[count] * 10,
                this_im_coords_sen[(count) + 1] * 10,
            ]
            av_coord = [(jun_ann[0] + sen_ann[0]) / 2, (jun_ann[1] + sen_ann[1]) / 2]
            print("jun %s, sen %s and av %s" % (jun_ann, sen_ann, av_coord))
            coordinates.append(
                av_coord
            )  # multiply by 10bevause they are downsacled by 10 for some reason in this file.
            count += 2
        inner_dict["coordinates"] = coordinates
        inner_dict["image"] = img_link

        if idx in valid_idx:
            data["validation"].append(inner_dict)
        else:
            data["training"].append(inner_dict)

    with open(
        output_path + "/w_valid/challenge_training.json", "w", encoding="utf-8"
    ) as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    # First test1 data
    test1_data = OrderedDict()
    test1_data["testing"] = []
    for idx in test1_idx:
        # print(idx)
        inner_dict = {}
        inner_dict["id"] = idx

        img_link = os.path.join(path_to_images, idx + ".nii.gz")
        # need to rotate these images to match the landmarks:
        this_im_coords_jun = anno_jun[int(idx) - 1]
        this_im_coords_sen = anno_sen[int(idx) - 1]

        coordinates = []
        count = (
            1  # start at 1 instead of 0 to ignore the first column indicating the index
        )
        for c in range(int(len(this_im_coords_jun) / 2)):
            jun_ann = [
                this_im_coords_jun[count] * 10,
                this_im_coords_sen[(count) + 1] * 10,
            ]
            sen_ann = [
                this_im_coords_sen[count] * 10,
                this_im_coords_sen[(count) + 1] * 10,
            ]
            av_coord = [(jun_ann[0] + sen_ann[0]) / 2, (jun_ann[1] + sen_ann[1]) / 2]
            print("jun %s, sen %s and av %s" % (jun_ann, sen_ann, av_coord))
            coordinates.append(
                av_coord
            )  # multiply by 10bevause they are downsacled by 10 for some reason in this file.
            count += 2
        inner_dict["coordinates"] = coordinates
        inner_dict["image"] = img_link

        test1_data["testing"].append(inner_dict)

    with open(
        output_path + "/w_valid/challenge_test1.json", "w", encoding="utf-8"
    ) as f:
        json.dump(test1_data, f, ensure_ascii=False, indent=4)
        # print(train_idx)

    # Now test2 data
    test2_data = OrderedDict()
    test2_data["testing"] = []
    for idx in test2_idx:
        # print(idx)
        inner_dict = {}
        inner_dict["id"] = idx

        img_link = os.path.join(path_to_images, idx + ".nii.gz")
        # need to rotate these images to match the landmarks:
        this_im_coords_jun = anno_jun[int(idx) - 1]
        this_im_coords_sen = anno_sen[int(idx) - 1]

        coordinates = []
        count = (
            1  # start at 1 instead of 0 to ignore the first column indicating the index
        )
        for c in range(int(len(this_im_coords_jun) / 2)):
            jun_ann = [
                this_im_coords_jun[count] * 10,
                this_im_coords_sen[(count) + 1] * 10,
            ]
            sen_ann = [
                this_im_coords_sen[count] * 10,
                this_im_coords_sen[(count) + 1] * 10,
            ]
            av_coord = [(jun_ann[0] + sen_ann[0]) / 2, (jun_ann[1] + sen_ann[1]) / 2]
            print("jun %s, sen %s and av %s" % (jun_ann, sen_ann, av_coord))
            coordinates.append(
                av_coord
            )  # multiply by 10bevause they are downsacled by 10 for some reason in this file.
            count += 2
        inner_dict["coordinates"] = coordinates
        inner_dict["image"] = img_link

        test2_data["testing"].append(inner_dict)
    with open(
        output_path + "/w_valid/challenge_test2.json", "w", encoding="utf-8"
    ) as f:
        json.dump(test2_data, f, ensure_ascii=False, indent=4)


# # transpose_all_nibs(os.path.join(root_path, path_to_images))
# # ISBI2015_to_json(path_to_jun_annotations, path_to_fold_infos, path_to_images, output_path, root_path)
# ISBI2015_to_json_with_vals(path_to_jun_annotations, path_to_fold_infos, path_to_images, output_path, root_path)


def ASPIRE_MEDIUM_to_JSON(
    path_to_annotations,
    path_to_images,
    output_path,
    modality,
    manual_omissions_uid,
    num_folds=5,
    debug=False,
):
    """
    Generate json from annotations of Medium ASPIRE dataset (~700 images for SA ~700 images for 4CH).
    For each fold: 80% is training, 20% is validation, 20% is testing.
    We generate a final "deployment json" where 10% of all images are validation, the rest is training.

    """
    assert modality in ["SA", "4ch"]
    anno = pd.read_csv(path_to_annotations)

    dataset_length = len(anno)
    # trainset_len = np.round(dataset_length*0.6)
    # testset_len = np.round(dataset_length*0.2)
    # validset_len = dataset_length - (trainset_len + testset_len)
    # print("dataet len is: ", dataset_length, "train, test and valid len are: ", trainset_len, testset_len, validset_len, ". added: ",
    #     trainset_len+validset_len+testset_len)

    indicies = np.arange(dataset_length)

    indicies_in_folds = np.array_split(indicies, num_folds - 1)

    for fold in range(num_folds):

        data = OrderedDict()

        # basic info
        data["name"] = "ASPIRE MEDIUM (~700)"
        data["description"] = (
            "Cardiac MRI "
            + modality
            + " View. Images 5 annotated landmarks from the ASPIRE MEDIUM dataset."
        )
        data["fold"] = fold

        data["training"] = []
        data["validation"] = []
        data["testing"] = []

        # DO the deployment fold here.
        if fold == (num_folds - 1):
            print("DEPLOYMENT FOLD")
            train_idx = list(
                np.random.choice(indicies, int(len(indicies) * 0.9), replace=False)
            )
            valid_idx = [x for x in indicies if x not in train_idx]
            test_idx = valid_idx
            print(train_idx)
            print(valid_idx)
            print(test_idx)
        else:
            test_fold = fold
            valid_fold = (test_fold + 1) % (num_folds - 1)
            train_folds = np.delete(np.arange(num_folds - 1), [test_fold, valid_fold])

            train_idx = [
                item
                for sublist in [indicies_in_folds[i] for i in train_folds]
                for item in sublist
            ]
            valid_idx = indicies_in_folds[valid_fold]
            test_idx = indicies_in_folds[test_fold]

        print(
            "\n",
            fold,
            "test fold: ",
            test_fold,
            "valid fold: ",
            valid_fold,
            "train folds: ",
            train_folds,
        )

        print(
            "dataet len is: ",
            dataset_length,
            "train, test and valid len are: ",
            len(train_idx),
            len(valid_idx),
            len(test_idx),
            ". added: ",
            len(train_idx) + len(valid_idx) + len(test_idx),
        )

        # Get all columns apart from patient ID (these represent all landamrk X and Y)
        all_columns = [x for x in anno.columns if x != "PatientID"]
        # Get rid of the _0 of _1 suffix and just get the name of each landmark
        split_cols_for_unique_lms = np.unique(
            [x.rpartition("_")[0] for x in all_columns]
        )

        for idx in indicies:

            uid = anno.iloc[[idx]]["PatientID"].values[0]
            if uid in manual_omissions_uid:
                continue
            # print(idx)
            inner_dict = {}
            inner_dict["id"] = uid

            relative_path = os.path.join(uid, "phase_1.dcm")
            img_link = os.path.join(path_to_images, relative_path)
            try:
                ds = dicom.dcmread(img_link).pixel_array
            except:
                try:
                    all_files = [
                        f
                        for f in listdir(os.path.join(path_to_images, uid))
                        if isfile(join(os.path.join(path_to_images, uid), f))
                    ]
                except:
                    print("could not find uid {}. skipping this patient.".format(uid))
                    continue

                min_phase_image = min(
                    all_files, key=lambda x: float((x.split("_")[1].split(".")[0]))
                )
                relative_path = os.path.join(uid, min_phase_image)
                img_link = os.path.join(path_to_images, relative_path)
                print(
                    "\n could not read dicom file: ",
                    os.path.join(path_to_images, uid, "phase_1.dcm"),
                    ". instead loading: ",
                    img_link,
                )

                ds = dicom.dcmread(img_link).pixel_array

            # **
            if debug:
                fig, ax = plt.subplots(1)
                ax.imshow(ds)
            # **

            # need to rotate these images to match the landmarks:
            coordinates = []
            for unique_lm in split_cols_for_unique_lms:
                x_coord = anno.iloc[[idx]][unique_lm + "_0"].values[0]
                y_coord = anno.iloc[[idx]][unique_lm + "_1"].values[0]

                # **
                if debug:

                    rect1 = patches.Rectangle(
                        (int(x_coord), int(y_coord)),
                        3,
                        3,
                        linewidth=2,
                        edgecolor="r",
                        facecolor="none",
                    )
                    ax.add_patch(rect1)
                # **

                coordinates.append([np.rint(x_coord), np.rint(y_coord)])

            if debug:
                print("\n uid: ", uid, "coordinates: ", coordinates)
                print("image link: ", img_link)
                if modality == "4ch" and uid in [
                    "PHD_6966",
                    "PHD_3366",
                    "PHD_3096",
                    "PHD_3063",
                    "PHD_1341",
                    "PHD_937",
                    "PHD_391",
                    "PHD_1034",
                    "PHD_3952",
                    "PHD_6966",
                    "PHD_960",
                    "PHD_3850",
                ]:
                    print("uid ", uid)
                    plt.show()
                plt.close()
                continue

            inner_dict["coordinates"] = coordinates
            inner_dict["image"] = relative_path

            if idx in valid_idx:
                data["validation"].append(inner_dict)
            if idx in test_idx:
                data["testing"].append(inner_dict)
            if idx in train_idx:
                data["training"].append(inner_dict)

                # print(train_idx)
        with open(
            output_path + "/fold" + str(fold) + ".json", "w", encoding="utf-8"
        ) as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


def ASPIRE_FOLLOWUP_to_JSON(
    path_to_annotations,
    path_to_images,
    output_path,
    modality,
    manual_omissions_uid,
    landmark_names,
    debug=False,
):
    """
    Data source: https://drive.google.com/drive/u/0/folders/1NBLdv7-ohcy23RyqTPpVS1YG65rjPcAh
    Generate json from annotations of followup ASPIRE dataset (~700 images for SA ~700 images for 4CH).
    For each fold: 80% is training, 20% is validation, 20% is testing.
    We generate a final "deployment json" where 10% of all images are validation, the rest is training.

    """

    # TODO: Get all the patients from the image, not all patients have an annotation in the image.

    assert modality in ["SA", "4ch"]
    anno = pd.read_excel(path_to_annotations, engine="openpyxl")

    dataset_length = len(anno)
    uids = anno["xnat_id"].unique()

    # NOw just get the uids with landmarks in this modality
    # all_dirs_image_folder = [x[0] for x in os.walk(path_to_images)]

    # Get all top level folders, these include all the patient uid

    filedepths1 = glob.glob(path_to_images + "*/*")
    all_dirs_top = list(filter(lambda f: os.path.isdir(f), filedepths1))

    # Need to search the variable files structures to find the DICOMS
    filesDepth3 = glob.glob(path_to_images + "*/*/*/*")
    all_dirs_image_folder = list(filter(lambda f: os.path.isdir(f), filesDepth3))
    print(
        "dataset len is: ",
        dataset_length,
        "uids are: ",
        len(uids),
        "and image directory length: ",
        len(all_dirs_image_folder),
    )

    # Sometimes the file struvtures are different for some, warn user and enforce they need to share the same structure.
    # (I have fixed the 2 cases that I found to be different.)
    for shallow in all_dirs_top:
        if len([x for x in all_dirs_image_folder if shallow in x]) == 0:
            assert ValueError(
                """ %s not in %s so will not be included. Make sure the former follows the directory structure of 
                 the rest images!
                """
                % (shallow, all_dirs_image_folder)
            )

    data = OrderedDict()

    # basic info
    data["name"] = "ASPIRE FOLLOWUP"
    data["description"] = (
        "Cardiac MRI "
        + modality
        + " View Images. 4 annotated landmarks from the ASPIRE FOLLOWUP dataset."
    )
    data["fold"] = 1

    data["testing"] = []

    not_found = []
    no_dicom = []

    for uid_ in uids:
        uid_ = str(uid_)
        if uid_ in manual_omissions_uid:
            continue

        print("UID ", uid_)
        # print(idx)
        inner_dict = {}
        inner_dict["id"] = uid_

        # get the landmarks coordinates for this uid
        # all_lm_coords = []

        # if the uid does not have the landamrk columns, it means it is the wrong modality, and move on.
        try:
            series_uid = anno.loc[
                ((anno["xnat_id"]).astype(str) == uid_)
                & (anno["LandMarkName"].isin(landmark_names)),
                "SeriesInstanceUID",
            ].unique()
            # print(series_uid)
            all_lm_coords = [[] for x in (series_uid)]
            # print("almcrd", all_lm_coords)
            for sidx, suid in enumerate(series_uid):
                # print(sidx, suid)
                for lm in landmark_names:
                    landmark_coords = [
                        np.round(
                            anno.loc[
                                ((anno["xnat_id"]).astype(str) == uid_)
                                & (anno["LandMarkName"] == lm)
                                & (anno["SeriesInstanceUID"] == suid),
                                "PointCoord[0]",
                            ].iloc[0]
                        ),
                        np.round(
                            anno.loc[
                                ((anno["xnat_id"]).astype(str) == uid_)
                                & (anno["LandMarkName"] == lm)
                                & (anno["SeriesInstanceUID"] == suid),
                                "PointCoord[1]",
                            ].iloc[0]
                        ),
                    ]
                    # print(uid_, lm, landmark_coords)

                    all_lm_coords[sidx][sidx].append(landmark_coords)
                all_lms = all_lm_coords[sidx][
                    0, 3, 2, 1
                ]  # for some reason need to rearrange to match other data.

        except:
            print("no landmarks found, wrong modality")
            continue

        # Get the exact phase info for the uid, the patient may have multiple scans with landmarks, so we will need to search through all.
        # print(series_uid)

        print("Number of scans in sheet:", len(series_uid))
        # print("length of landmarks ", len(all_lm_coords))
        # for x in range(len(all_lm_coords)):
        #     print(len(all_lm_coords[x]), end=",")
        # print()
        # continue

        # Search through the directory list to find the image with matching uid.
        correct_dir = [x for x in all_dirs_image_folder if uid_ in x]

        if len(correct_dir) == 0:
            print("NOT FOUND: ", uid_)
            not_found.append(uid_)
            continue

        # Patients may have more than 1 scan.
        elif len(correct_dir) > 1:
            candidate_uids = []
            if "PHD" in uid_:
                for dir_name in correct_dir:
                    uid_here = dir_name.split("/")[-3]
                    print("candidate dirname: ", dir_name, "uid_here: ", uid_here)
                    if "-" in uid_here:
                        candidate_uids.append(uid_here.split("-")[0])
                    else:
                        candidate_uids.append(uid_here)
            else:
                uid_here = dir_name.split("/")[-3]
                print("candidate dirname: ", dir_name, "uid_here: ", uid_here)
                if "_" in uid_here:
                    candidate_uids.append(dir_name.split("_")[0])
                else:
                    candidate_uids.append(dir_name)
            print("candidate uids: ", candidate_uids)

            found_matching = False
            for cuid_idx, cuid in enumerate(candidate_uids):
                if cuid == uid_:
                    correct_dir = correct_dir[cuid_idx]
                    found_matching = True
                    break
            if found_matching == False:
                print("could not find dir for uid %s in %s" % (uid_, correct_dir))
                not_found.append(uid_)
                continue
        else:
            correct_dir = correct_dir[0]

        print("found the matching directory: ", correct_dir)

        path_to_dicoms = os.path.join(path_to_images, correct_dir)
        filedepths1 = glob.glob(path_to_dicoms + "/*")
        potential_dicoms = list(filter(lambda f: os.path.isfile(f), filedepths1))

        # print("Potentail dicoms: ", potential_dicoms)
        # Go through matching dicoms and match the phase with the series_uid. Then get phase with earliest InstanceNumber
        found_dicom = False
        matching_dicoms = []
        # matching_dicom_paths = []
        # se_uid_match_idx = []
        for dcm in potential_dicoms:
            ds = dicom.dcmread(os.path.join(path_to_dicoms, dcm))
            for se_uid_idx, se_uid in enumerate(series_uid):
                if ds["SeriesInstanceUID"].value == se_uid:
                    found_dicom = True

                    dcm_path = os.path.join(os.path.join(path_to_dicoms, dcm))
                    setattr(ds, "FilePath", dcm_path)
                    setattr(ds, "Landmarks", all_lm_coords[se_uid_idx])

                    matching_dicoms.append(ds)

                    # print("found matching dicom: ", img_link)

                    # break
        if found_dicom == False:
            no_dicom.append(uid_)
            print(
                "could not find matching dicom of %s in %s"
                % (series_uid, potential_dicoms)
            )
            continue
        else:
            matching_dicoms.sort(key=lambda x: x.InstanceNumber, reverse=False)
            earliest_phase = matching_dicoms[0]

        inner_dict["coordinates"] = earliest_phase.Landmarks
        inner_dict["image"] = earliest_phase.FilePath
        inner_dict["modality"] = modality
        inner_dict["landmark_names"] = landmark_names

        data["testing"].append(inner_dict)

        if debug:
            print("length of matching dicoms: ", len(matching_dicoms))
            print("earliest phase dicom: ", earliest_phase)
            print("inner dict: ", inner_dict)
            lm_coords = inner_dict["coordinates"]
            fig, ax = plt.subplots(1)
            ax.imshow(ds.pixel_array)
            for lm in lm_coords:
                rect1 = patches.Rectangle(
                    (int(lm[0]), int(lm[1])),
                    3,
                    3,
                    linewidth=2,
                    edgecolor="r",
                    facecolor="none",
                )
                ax.add_patch(rect1)
            plt.show()
            plt.close()

        print()

    print("no dicom: ", no_dicom)
    print("not found: ", not_found)

    os.makedirs(output_path, exist_ok=True)

    with open(output_path + "/fold0.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def ASPIRE_FOLLOWUP_to_JSON_careful(
    path_to_annotations,
    path_to_images,
    output_path,
    modality,
    manual_omissions_uid,
    landmark_names,
    debug=False,
):
    """
    Data source: https://drive.google.com/drive/u/0/folders/1NBLdv7-ohcy23RyqTPpVS1YG65rjPcAh
    Generate json from annotations of followup ASPIRE dataset (~700 images for SA ~700 images for 4CH).
    For each fold: 80% is training, 20% is validation, 20% is testing.
    We generate a final "deployment json" where 10% of all images are validation, the rest is training.

    """

    # TODO: Get all the patients from the image, not all patients have an annotation in the image.

    assert modality in ["SA", "4ch"]
    anno = pd.read_excel(path_to_annotations, engine="openpyxl")

    dataset_length = len(anno)
    uids = anno["xnat_id"].unique()

    data = OrderedDict()

    # basic info
    data["name"] = "ASPIRE FOLLOWUP"
    data["description"] = (
        "Cardiac MRI "
        + modality
        + " View Images. 4 annotated landmarks from the ASPIRE FOLLOWUP dataset."
    )
    data["fold"] = 1

    data["testing"] = []

    no_matching_suid = []
    no_landmarks_for_suid = []

    # Get all folders in directory
    if type(path_to_images) == list:
        all_folders = []
        [all_folders + glob.glob(x + "/*") for x in path_to_images]

    else:
        all_folders = glob.glob(path_to_images + "/*")
    print("all folders: ", len(all_folders))

    # print("len of all dcms,", len(all_dcms))
    # exit()

    for folder in all_folders:
        print("\n \n Loading: ", folder)

        # Get all dicoms from folder, and get the earliest phase (InstanceNumber)
        dcm_phases = []  # list of dicom dataset objects (phases)
        phase_files = glob.glob(folder + "/**/*.dcm", recursive=True)
        for phase_file in phase_files:
            dataset = dicom.dcmread(phase_file)
            setattr(dataset, "file_path", phase_file)
            dcm_phases.append(dataset)

        # Sort by instance number to get earliest phase
        dcm_phases.sort(key=lambda x: x.InstanceNumber, reverse=False)
        ep_dcm = dcm_phases[0]

        suid = ep_dcm.SeriesInstanceUID
        xnat_id = ep_dcm.PatientID

        inner_dict = {}
        inner_dict["id"] = xnat_id
        inner_dict["suid"] = suid
        inner_dict["image"] = ep_dcm.file_path
        inner_dict["modality"] = modality
        inner_dict["landmark_names"] = landmark_names

        matching_suids = anno[anno["SeriesInstanceUID"] == suid]

        # print("aaa", matching_suids["LandMarkName"].values)

        if len(matching_suids) == 0:
            print("could not find matching suid: ", suid, " for xnat_id: ", xnat_id)
            no_matching_suid.append([xnat_id, suid])
            inner_dict["has_annotation"] = False
            inner_dict["coordinates"] = None
            data["testing"].append(inner_dict)
            continue
        elif not all(
            elem in matching_suids["LandMarkName"].values for elem in landmark_names
        ):
            print("could not find landmarks for: ", suid, " for xnat_id: ", xnat_id)
            no_landmarks_for_suid.append([xnat_id, suid])
            inner_dict["has_annotation"] = False
            inner_dict["coordinates"] = None
            data["testing"].append(inner_dict)
            continue
        else:
            inner_dict["has_annotation"] = True

        print("rows with matching suid: ", (matching_suids))
        # if not matching_suids.LandmarkName.values.isin(landmark_names):
        #     continue
        all_lms = []
        # match series uid with landmarking sheet and extract landmarks
        for lm in landmark_names:
            landmark_coords = [
                np.round(
                    matching_suids.loc[
                        (anno["LandMarkName"] == lm), "PointCoord[0]"
                    ].iloc[0]
                ),
                np.round(
                    matching_suids.loc[
                        (anno["LandMarkName"] == lm), "PointCoord[1]"
                    ].iloc[0]
                ),
            ]
            all_lms.append(landmark_coords)

        if modality == "SA":  # for some reason need to rearrange to match other data.
            new_order = [0, 3, 2, 1]
            all_lms = [all_lms[i] for i in new_order]

        inner_dict["coordinates"] = all_lms
        data["testing"].append(inner_dict)

        if debug:
            print("earliest phase dicom: ", ep_dcm)
            print("inner dict: ", inner_dict)
            lm_coords = inner_dict["coordinates"]
            fig, ax = plt.subplots(1)
            ax.imshow(ep_dcm.pixel_array)
            for lm in lm_coords:
                rect1 = patches.Rectangle(
                    (int(lm[0]), int(lm[1])),
                    3,
                    3,
                    linewidth=2,
                    edgecolor="r",
                    facecolor="none",
                )
                ax.add_patch(rect1)
            plt.show()
            plt.close()

            print()

    print("no matching suid: ", no_matching_suid)
    print("no landmarks for suid: ", no_landmarks_for_suid)
    print("number with no matching suid ", len(no_matching_suid))
    print("number with no landamrks ", len(no_landmarks_for_suid))

    os.makedirs(output_path, exist_ok=True)

    with open(output_path + "/fold0.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def ASPIRE_LARGE_no_annotations(
    path_to_images, output_path, modality, manual_omissions_uid, debug=False
):
    """
    Data source: https://drive.google.com/drive/folders/1NKM87o8gkHmrPuQMOYVo_3-Dv0kgas69
    Generate json from annotations of followup ASPIRE dataset (~unknown images for SA ~unknown images for 4CH).
    For each fold: 100% is testing.
    We have no annotations.
    """

    assert modality in ["SA", "4ch", "4C"]

    data = OrderedDict()

    # basic info
    data["name"] = "ASPIRE LARGE"
    data["description"] = "Cardiac MRI " + modality + " View Images."
    data["fold"] = 1

    data["testing"] = []

    # Get all folders in directory
    if type(path_to_images) == list:
        all_folders = []
        for x in path_to_images:
            all_folders += glob.glob(x + "/*")

    else:
        all_folders = glob.glob(path_to_images + "/*")
    print("all folders: ", len(all_folders))

    # print("len of all dcms,", len(all_dcms))
    # exit()

    for folder in all_folders:
        print("\n \n Loading: ", folder)

        # Get all dicoms from folder, and get the earliest phase (InstanceNumber)
        dcm_phases = []  # list of dicom dataset objects (phases)
        phase_files = glob.glob(folder + "/**/*.dcm", recursive=True)
        for phase_file in phase_files:
            dataset = dicom.dcmread(phase_file)
            setattr(dataset, "file_path", phase_file)
            dcm_phases.append(dataset)

        # Sort by instance number to get earliest phase
        dcm_phases.sort(key=lambda x: x.InstanceNumber, reverse=False)
        ep_dcm = dcm_phases[0]

        suid = ep_dcm.SeriesInstanceUID
        xnat_id = ep_dcm.PatientID

        inner_dict = {}
        inner_dict["patient_id"] = xnat_id
        inner_dict["id"] = suid
        inner_dict["suid"] = suid

        inner_dict["image"] = ep_dcm.file_path
        inner_dict["modality"] = modality
        inner_dict["has_annotation"] = False
        inner_dict["coordinates"] = None
        data["testing"].append(inner_dict)

    os.makedirs(output_path, exist_ok=True)

    with open(output_path + "/fold0.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def combine_aspire_followup_to_train(
    path_to_baseline,
    path_to_followup,
    baseline_pre,
    followup_pre,
    output_path,
    modality,
    num_folds=10,
    debug=False,
):
    """
    Combine the small ASPIRE dataset with the followup ASPIRE dataset.
    Make one json that is: 80% is training, 20% is validation.
    Make 10 folds that are: 90% is training, 10% is validation. We can make an ensemble of these.

    """

    def map_to_obj(x, y):
        x["image"] = os.path.join(y, x["image"])
        return x

    # Load small ASPIRE
    with open(path_to_baseline) as json_file:
        baseline_aspire = json.load(json_file)

    # Load followup ASPIRE
    with open(path_to_followup) as json_file:
        followup_aspire = json.load(json_file)

    baseline_aspire = [
        [x for x in (map_to_obj(x, baseline_pre) for x in baseline_aspire[split])]
        for split in ["training", "validation", "testing"]
    ]
    baseline_aspire = [item for sublist in baseline_aspire for item in sublist]

    followup_aspire = [
        x for x in (map_to_obj(x, followup_pre) for x in followup_aspire["testing"])
    ]
    # Combine
    combined = baseline_aspire
    print("length of baseline: ", len(combined))

    combined_w_coords = combined + [
        x for x in followup_aspire if x["has_annotation"] == True
    ]

    combined_all = combined + followup_aspire

    print("combined with coords length: ", len(combined_w_coords))
    print("combined all length: ", len(combined_all))

    dataset_length = len(combined_w_coords)
    indicies = np.arange(dataset_length)
    indicies_in_folds = np.array_split(indicies, num_folds)

    # 1) make a fold that is 80% training, 20% validation
    data = OrderedDict()

    # basic info
    data["name"] = "ASPIRE BASELINE + FOLLOWUP"
    data["description"] = "Cardiac MRI " + modality + " View Images."
    data["fold"] = 99

    data["training"] = []
    data["validation"] = []
    data["testing"] = []

    train_idx = list(
        np.random.choice(indicies, int(len(indicies) * 0.8), replace=False)
    )
    valid_idx = [x for x in indicies if x not in train_idx]

    data["training"] = [combined_w_coords[x] for x in train_idx]
    data["validation"] = [combined_w_coords[x] for x in valid_idx]

    print("lednth of train and val: ", len(data["training"]), len(data["validation"]))

    os.makedirs(output_path, exist_ok=True)

    with open(output_path + "/fold99.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    # 2) make 10 folds that are 90% training, 10% validation
    for fold in range(num_folds):
        data = OrderedDict()

        # basic info
        data["name"] = "ASPIRE BASELINE + FOLLOWUP"
        data["description"] = "Cardiac MRI " + modality + " View Images."
        data["fold"] = fold

        data["training"] = []
        data["validation"] = []
        data["testing"] = []

        valid_fold = fold
        train_folds = np.delete(np.arange(num_folds), [valid_fold])

        print("train folds %s, valid fold %s" % (train_folds, valid_fold))

        train_idx = [
            item
            for sublist in [indicies_in_folds[i] for i in train_folds]
            for item in sublist
        ]
        valid_idx = indicies_in_folds[valid_fold]

        data["training"] = [combined_w_coords[x] for x in train_idx]
        data["validation"] = [combined_w_coords[x] for x in valid_idx]

        print(
            "length of train and val: ", len(data["training"]), len(data["validation"])
        )

        with open(
            output_path + "/fold" + str(fold) + ".json", "w", encoding="utf-8"
        ) as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


local = True
if local:
    prestring = "/mnt/tale_shared/"
else:
    prestring = "/shared/tale2/Shared/"


def aspire_followup_gen():
    for modality in ["4ch"]:  # [4ch, "SA"]
        if modality == "SA":
            manual_omissions = []
            path_to_images = (
                prestring
                + "data/CMRI/PAH-Followup/SA - Follow-up-20220817T130121Z-001/SA - Follow-up"
            )
            landmarks = [
                "SA_ED_Inferior_Insertion",
                "SA_ED_Superior_Insertion",
                "SA_ED_RV_Free_Wall_Inflection",
                "SA_ED_Mid_LV_Lateral_Wall",
            ]
        else:
            path_to_images = (
                prestring
                + "data/CMRI/PAH-Followup/4CH - Follow-up-20220817T130118Z-001/4CH - Follow-up"
            )
            # path_to_images = "/mnt/tale_shared/data/CMRI/PAH-Followup/4CH - Initial-20220817T155225Z-001/4CH - Initial"
            manual_omissions = []
            landmarks = [
                "4CH_ED_LV_Apex",
                "4CH_ED_Lateral_Mitral_Annulus",
                "4CH_ED_Lateral_Tricuspid_Annulus",
                "4CH_ED_Spinal_Cord",
            ]

        path_to_annotations = (
            prestring + "data/CMRI/PAH-Followup/all_landmarks_from_MIMS.xlsx"
        )
        output_path = (
            prestring
            + "data/CMRI/PAH-Followup/landmark_localisation_annotations/"
            + modality
        )

        ASPIRE_FOLLOWUP_to_JSON_careful(
            path_to_annotations,
            path_to_images,
            output_path,
            modality,
            manual_omissions,
            landmarks,
            debug=False,
        )


def aspire_large_gen():
    for modality in ["SA", "4ch"]:  # [4ch, "SA"]
        if modality == "SA":
            manual_omissions = []
            path_to_images = [
                prestring + "data/CMRI/PAH-Large/sa 1",
                prestring + "data/CMRI/PAH-Large/sa 2",
                prestring + "data/CMRI/PAH-Large/sa 3",
            ]
        else:
            path_to_images = [
                prestring + "data/CMRI/PAH-Large/4ch 1",
                prestring + "data/CMRI/PAH-Large/4ch 2",
                prestring + "data/CMRI/PAH-Large/4ch 3",
            ]
            manual_omissions = []

        output_path = (
            prestring
            + "data/CMRI/PAH-Large/landmark_localisation_annotations/"
            + modality
        )

        ASPIRE_LARGE_no_annotations(
            path_to_images, output_path, modality, manual_omissions, debug=False
        )


def combine_gen():
    for modality in ["SA", "4ch"]:  # [4ch, "SA"]
        json_followup = (
            prestring
            + "data/CMRI/PAH-Followup/landmark_localisation_annotations_resized/"
            + modality
            + "/fold0.json"
        )
        json_baseline = (
            prestring
            + "data/CMRI/PAH-Baseline/Proc/landmark_localisation_annotations_npz/"
            + modality
            + "/fold0.json"
        )
        output_path = (
            prestring
            + "data/CMRI/PAH-Followup/landmark_localisation_annotations_combined_npz/"
            + modality
        )

        baseline_pre = "PAH-Baseline/Proc/" + modality + "/"

        # output_path =prestring+"data/CMRI/PAH-Followup/landmark_localisation_annotations_combined_resized/"+ modality

        if modality == "4ch":
            # followup_pre = "PAH-Followup/"  + "/4CH - Follow-up-20220817T130118Z-001/4CH - Follow-up/"
            followup_pre = "PAH-Followup/4CH_resized/"

        else:
            # followup_pre = "PAH-Followup/"  + "/SA - Follow-up-20220817T130121Z-001/SA - Follow-up/"
            followup_pre = "PAH-Followup/SA_resized/"

        combine_aspire_followup_to_train(
            json_baseline,
            json_followup,
            baseline_pre,
            followup_pre,
            output_path,
            modality,
            debug=False,
        )


# path_to_jun_annotations= "/mnt/tale_shared/schobs/data/ISBI2015_landmarks/setup_ann/all_landmarks/all_junior.csv"
# # path_to_sen_annotations= "/mnt/tale_shared/schobs/data/ISBI2015_landmarks/setup_ann/all_landmarks/all_senior.csv"
# path_to_train_idx = "/mnt/tale_shared/schobs/data/ISBI2015_landmarks/setup_ann/all_landmarks/train.txt"
# path_to_test1_idx = "/mnt/tale_shared/schobs/data/ISBI2015_landmarks/setup_ann/all_landmarks/test1.txt"
# path_to_test2_idx = "/mnt/tale_shared/schobs/data/ISBI2015_landmarks/setup_ann/all_landmarks/test2.txt"

# path_to_images =  "images/"
# output_path = "/mnt/tale_shared/schobs/data/ISBI2015_landmarks/lann_folds/"
# # ISBI2015_to_json_with_vals_challenge(path_to_jun_annotations, path_to_sen_annotations, path_to_train_idx, path_to_test1_idx, path_to_test2_idx, path_to_images, output_path, val_size=0.1)
# path_to_fold_infos = "/mnt/tale_shared/schobs/data/ISBI2015_landmarks/setup_ann/all_landmarks/cv/"

# ISBI2015_to_json_with_vals_vary_train(path_to_jun_annotations, path_to_fold_infos, path_to_images, output_path, val_size=0.1,num_folds=4)

# aspire_followup_gen()
# combine_gen()
# aspire_large_gen()


#     output_path ="/mnt/tale_shared/data/CMRI/PAH-Baseline/Proc/landmark_localisation_annotations/"+ modality
#     ASPIRE_MEDIUM_to_JSON(path_to_annotations, path_to_images, output_path, modality, manual_omissions, num_folds=6, debug=False )

# for modality in ["4ch", "SA"]:
#     if modality == "SA":
#         manual_omissions = ["PHD_3224", "PHD_873", "PHD_948", "PHD_5614", "PHD_1526", "PHD_3865", "PHD_206"]  #check obsidian [[Landmarking APSIRE 700]]  for why omitted
#     else:
#         manual_omissions = ["PHD_6966", "PHD_1421", "PHD_3237", "PHD_3224"] #check obsidian [[Landmarking APSIRE 700]]  for why omitted
#     path_to_annotations = os.path.join("/mnt/tale_shared/data/CMRI/PAH-Baseline/Proc/", modality+"_landmark_flip.csv")
#     path_to_images =  os.path.join("/mnt/tale_shared/data/CMRI/PAH-Baseline/Proc/", modality)
#     output_path ="/mnt/tale_shared/data/CMRI/PAH-Baseline/Proc/landmark_localisation_annotations/"+ modality
#     ASPIRE_MEDIUM_to_JSON(path_to_annotations, path_to_images, output_path, modality, manual_omissions, num_folds=6, debug=False )


# April 2023 for prasun

for modality in ["4C", "SA"]:
    if modality == "SA":
        manual_omissions = []
    else:
        manual_omissions = []
    path_to_images = os.path.join("/mnt/tale_shared/data/CMRI/new scans", modality)
    output_path = os.path.join("/mnt/tale_shared/data/CMRI/new scans", modality+"recipe")

    ASPIRE_LARGE_no_annotations(path_to_images, output_path, modality, manual_omissions, debug=False)
    # ASPIRE_MEDIUM_to_JSON(path_to_annotations, path_to_images, output_path, modality, manual_omissions, num_folds=6, debug=False )
