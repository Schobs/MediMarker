import copy
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
import json


def ASPIRE_FOLLOWUP_resize(
    path_to_annotations,
    path_to_images,
    output_path_images,
    output_path_anno,
    modality,
    manual_omissions_uid,
    landmark_names,
    resize=[512, 512],
    debug=False,
):

    """
    Data source: https://drive.google.com/drive/u/0/folders/1NBLdv7-ohcy23RyqTPpVS1YG65rjPcAh
    Resizes the images to 512x512 and saves them in the output path. Also resize annotations

    """

    # TODO: Get all the patients from the image, not all patients have an annotation in the image.

    assert modality in ["SA", "4ch"]
    anno = pd.read_excel(path_to_annotations, engine="openpyxl")

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

        # Basic info
        suid = ep_dcm.SeriesInstanceUID
        xnat_id = ep_dcm.PatientID
        inner_dict = {}
        inner_dict["id"] = xnat_id
        inner_dict["suid"] = suid
        inner_dict["modality"] = modality
        inner_dict["landmark_names"] = landmark_names

        matching_suids = anno[anno["SeriesInstanceUID"] == suid]

        # If image does not have landmarks in spreadsheet, just save the image with None coords.
        if len(matching_suids) == 0:
            print("could not find matching suid: ", suid, " for xnat_id: ", xnat_id)
            no_matching_suid.append([xnat_id, suid])
            inner_dict["has_annotation"] = False
            inner_dict["coordinates"] = None
            # continue import json
        elif not all(
            elem in matching_suids["LandMarkName"].values for elem in landmark_names
        ):
            print("could not find landmarks for: ", suid, " for xnat_id: ", xnat_id)
            no_landmarks_for_suid.append([xnat_id, suid])
            inner_dict["has_annotation"] = False
            inner_dict["coordinates"] = None
            # continue
        else:
            # If there were coords,
            inner_dict["has_annotation"] = True

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

            if (
                modality == "SA"
            ):  # for some reason need to rearrange to match other data.
                new_order = [0, 3, 2, 1]
                all_lms = [all_lms[i] for i in new_order]

            inner_dict["coordinates"] = all_lms

        # Potential resizings if image does not match desired size
        ep_dcm_image = ep_dcm.pixel_array
        need_resize = False
        if ep_dcm_image.shape[0] != resize[0] or ep_dcm_image.shape[1] != resize[1]:

            need_resize = True

            downscale_factor = [
                ep_dcm_image.shape[0] / resize[0],
                ep_dcm_image.shape[1] / resize[1],
            ]
            print("downscale factor: ", downscale_factor)
            ep_dcm_image_resize = np.array(Image.fromarray(ep_dcm_image).resize(resize))

            old_ep = copy.deepcopy(ep_dcm_image)
            ep_dcm_image = ep_dcm_image_resize

            # IF there are annotations, update them
            if inner_dict["has_annotation"]:
                resized_lms = []
                print("og landmarks ", all_lms)
                for lm in all_lms:
                    resized_lms.append(
                        [lm[0] / downscale_factor[1], lm[1] / downscale_factor[0]]
                    )

                print("resized lms ", resized_lms)
                old_lms = copy.deepcopy(all_lms)
                all_lms = resized_lms

                # update the dictionary.
                inner_dict["coordinates"] = all_lms

        # Add new image path to json, we are saving ALL as an npz.

        this_im_path = ep_dcm.file_path
        print("the path to the image: ", this_im_path)
        inner_dict["image"] = (
            (this_im_path.split("Follow-up")[-1]).split(".dcm")[0] + ".npz"
        )[1:]

        print("what im sabving currently", inner_dict["image"])
        # exit()
        data["testing"].append(inner_dict)
        print(
            "path adding to json: ",
            ((this_im_path.split("Follow-up")[-1]).split(".dcm")[0] + ".npz")[1:],
        )

        # save the new dicom

        save_path_this_im = (
            output_path_images
            + (this_im_path.split("Follow-up")[-1]).split(".dcm")[0]
            + ".npz"
        )
        make_this_dir = "/".join(save_path_this_im.split("/")[:-1])
        print("save path: ", save_path_this_im, "Make dir: ", make_this_dir)

        os.makedirs(make_this_dir, exist_ok=True)

        np.savez(save_path_this_im, ep_dcm_image)

        if debug and need_resize:
            # print("earliest phase dicom: ", ep_dcm)
            # print("inner dict: ", inner_dict)
            lm_coords = inner_dict["coordinates"]
            fig, ax = plt.subplots(2, 1)
            ax[0].imshow(old_ep)
            ax[1].imshow(ep_dcm_image)

            if inner_dict["has_annotation"]:
                for lm in old_lms:
                    rect1 = patches.Rectangle(
                        (int(lm[0]), int(lm[1])),
                        3,
                        3,
                        linewidth=2,
                        edgecolor="r",
                        facecolor="none",
                    )
                    ax[0].add_patch(rect1)

                for lm in all_lms:
                    rect1 = patches.Rectangle(
                        (int(lm[0]), int(lm[1])),
                        3,
                        3,
                        linewidth=2,
                        edgecolor="m",
                        facecolor="none",
                    )
                    ax[1].add_patch(rect1)
            plt.show()
            plt.close()

            print()

    print("no matching suid: ", no_matching_suid)
    print("no landmarks for suid: ", no_landmarks_for_suid)
    print("number with no matching suid ", len(no_matching_suid))
    print("number with no landamrks ", len(no_landmarks_for_suid))

    os.makedirs(output_path_anno, exist_ok=True)

    with open(output_path_anno + "/fold0.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# (path_to_annotations,root_path, output_path_anno, modality, debug=True)


def dicom_to_npz(
    path_to_annotations,
    root_path,
    output_path_anno,
    modality,
    resize=[512, 512],
    debug=False,
):

    """
    Data source: https://drive.google.com/drive/u/0/folders/1NBLdv7-ohcy23RyqTPpVS1YG65rjPcAh
    Changes .dcm to .npz.

    """

    assert modality in ["SA", "4ch"]

    print("path to ann", path_to_annotations)
    all_anno_files = glob.glob(path_to_annotations + "/*")
    print("all anno files", all_anno_files)

    for anno_path in all_anno_files:
        if ".json" not in anno_path:
            continue

        print("anno path: ", anno_path)
        with open(anno_path, "r") as j:
            anno = json.loads(j.read())

        anno_split_name = anno_path.split("/")[-1]
        for split in ["training", "validation", "testing"]:

            for sample in anno[split]:

                image = dicom.dcmread(
                    os.path.join(root_path, sample["image"])
                ).pixel_array

                print("image shape ", image.shape)

                if image.shape[0] != resize[0] or image.shape[1] != resize[1]:
                    print(sample, "wrong size, resizing", image.shape[0])

                    # save the new dicom

                save_path_this_im = os.path.join(
                    root_path, sample["image"].split(".dcm")[0] + ".npz"
                )
                # make_this_dir = '/'.join(save_path_this_im.split("/")[:-1])
                print("save path: ", save_path_this_im)

                np.savez(save_path_this_im, image)
                sample["image"] = sample["image"].split(".dcm")[0] + ".npz"

        os.makedirs(output_path_anno, exist_ok=True)

        save_anno_to = os.path.join(output_path_anno, anno_split_name)
        print("saving to ", save_anno_to)
        with open(save_anno_to, "w", encoding="utf-8") as f:
            json.dump(anno, f, ensure_ascii=False, indent=4)


local = True
if local:
    prestring = "/mnt/tale_shared/"
else:
    prestring = "/shared/tale2/Shared/"


def resize_aspire_followup():
    for modality in ["4ch", "SA"]:  # [4ch, "SA"]
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
            output_path_images = prestring + "data/CMRI/PAH-Followup/SA_resized"

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
            output_path_images = prestring + "data/CMRI/PAH-Followup/4CH_resized"

        path_to_annotations = (
            prestring + "data/CMRI/PAH-Followup/all_landmarks_from_MIMS.xlsx"
        )
        output_path_anno = (
            prestring
            + "data/CMRI/PAH-Followup/landmark_localisation_annotations_resized/"
            + modality
        )

        ASPIRE_FOLLOWUP_resize(
            path_to_annotations,
            path_to_images,
            output_path_images,
            output_path_anno,
            modality,
            manual_omissions,
            landmarks,
            resize=[512, 512],
            debug=False,
        )


# resize_aspire_followup()


def aspire_dicom_to_npz():
    for modality in ["4ch", "SA"]:  # [4ch, "SA"]
        # if modality == "SA":
        #     manual_omissions = []
        #     path_to_images = prestring+"data/CMRI/PAH-Followup/SA - Follow-up-20220817T130121Z-001/SA - Follow-up"
        #     landmarks = ["SA_ED_Inferior_Insertion","SA_ED_Superior_Insertion", "SA_ED_RV_Free_Wall_Inflection", "SA_ED_Mid_LV_Lateral_Wall" ]
        #     output_path_images =prestring+"data/CMRI/PAH-Followup/SA_resized"

        # else:
        #     path_to_images = prestring+"data/CMRI/PAH-Followup/4CH - Follow-up-20220817T130118Z-001/4CH - Follow-up"
        #     # path_to_images = "/mnt/tale_shared/data/CMRI/PAH-Followup/4CH - Initial-20220817T155225Z-001/4CH - Initial"
        #     manual_omissions = []
        #     landmarks = ["4CH_ED_LV_Apex", "4CH_ED_Lateral_Mitral_Annulus","4CH_ED_Lateral_Tricuspid_Annulus", "4CH_ED_Spinal_Cord"]
        #     output_path_images = prestring+"data/CMRI/PAH-Followup/4CH_resized"

        path_to_annotations = (
            prestring
            + "data/CMRI/PAH-Baseline/Proc/landmark_localisation_annotations/"
            + modality
        )
        output_path_anno = (
            prestring
            + "data/CMRI/PAH-Baseline/Proc/landmark_localisation_annotations_npz/"
            + modality
        )
        root_path = prestring + "data/CMRI/PAH-Baseline/Proc/" + modality
        dicom_to_npz(
            path_to_annotations, root_path, output_path_anno, modality, debug=True
        )


# aspire_dicom_to_npz()

resize_aspire_followup()
