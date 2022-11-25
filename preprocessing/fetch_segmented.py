import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import shutil

INPUT_DIR = "/home/ciganche/Documents/AI_challange/data/raw/test/proj-60a14ca503bcad0ad27cada9/"
OUTPUT_DIR = "/home/ciganche/Documents/AI_challange/data/preprocessed/testic/"
SEGMENTATION_DIR_PREFIX = "dt-neuro-parcellation-volume"
IMAGE_EXTENSION = ".nii.gz"
OUTPUT_EXTENSION = "_segmented.nii.gz"

#%%

def get_dir_names(directory, seg_prefix):
    seg_dir = ""
    for f in os.scandir(directory):
        if seg_prefix in f.path:
            seg_dir = f.path
    return seg_dir


def get_nii(directory, extension):
    image_path = ""
    for f in os.scandir(directory):
        if extension in f.path:
            image_path = f.path
    return image_path


sample_directories = [f.path for f in os.scandir(INPUT_DIR) if f.is_dir()]


for sample_dir in sample_directories:
    # 0: t1, 1: mask dir
    segmentation_dir = get_dir_names(sample_dir, SEGMENTATION_DIR_PREFIX)
    if segmentation_dir == "":
        print("No segmentation directories for: " + sample_dir)
        continue


    segmentation_file = get_nii(segmentation_dir, IMAGE_EXTENSION)
    if segmentation_file == "":
        print("No segmentation file found for: " + sample_dir)

    new_name = OUTPUT_DIR + segmentation_file.split("/")[-3] + OUTPUT_EXTENSION
    print(new_name)
    shutil.copy(segmentation_file, new_name)
