import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

INPUT_DIR = "/home/ciganche/Documents/AI_challange/data/raw/test/proj-60a14ca503bcad0ad27cada9/"
OUTPUT_DIR = "/home/ciganche/Documents/AI_challange/data/preprocessed/test/"

T1_DIR_SUFFIX = "dt-neuro-anat-t1"
MASK_DIR_SUFFIX = "dt-neuro-mask"
IMAGE_EXTENSION = ".nii.gz"

def get_dir_names(directory, t1_suffix, mask_suffix):
    t1_dir = ""
    mask_dir = ""
    for f in os.scandir(directory):
        if t1_suffix in f.path:
            t1_dir = f.path
        if mask_suffix in f.path:
            mask_dir = f.path

    return t1_dir, mask_dir

def get_nii(directory, extension):
    image_path = ""
    for f in os.scandir(directory):
        if extension in f.path:
            image_path = f.path

    return image_path

def standardize(image, a, b):
    mmax = np.max(image)
    mmin = np.min(image)
    return np.round(a + (((image - mmin)*(b-a))/(mmax - mmin)), 7)

sample_directories = [f.path for f in os.scandir(INPUT_DIR) if f.is_dir()]

for sample_dir in sample_directories:
    # 0: t1, 1: mask dir
    sample_tuple = get_dir_names(sample_dir, T1_DIR_SUFFIX, MASK_DIR_SUFFIX)

    if sample_tuple[0] == "" or sample_tuple[1] == "":
        print("No t1/mask directories for: " + sample_dir)
        continue

    t1_nii_path = get_nii(sample_tuple[0], IMAGE_EXTENSION)
    mask_nii_path = get_nii(sample_tuple[1], IMAGE_EXTENSION)

    if t1_nii_path == "" or mask_nii_path == "":
        print("No t1/mask found for: " + sample_dir)

    t1_nii = nib.load(t1_nii_path)
    mask_nii = nib.load(mask_nii_path)
    masked_t1 = t1_nii.get_fdata() * mask_nii.get_fdata()
    normalized_masked_t1 = standardize(masked_t1, 0, 10)
    # Use the same affine of T1 image when constructing the new
    ret_val_nii = nib.Nifti1Image(normalized_masked_t1, affine=t1_nii.affine, header=t1_nii.header)
    name = sample_dir.split("/")[-1] + ".nii"
    nib.save(ret_val_nii, OUTPUT_DIR+name)
