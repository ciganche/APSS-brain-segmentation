import pandas as pd
import numpy as np
import nibabel as nib
import os
import skimage.transform
from scipy.ndimage import binary_dilation

# input directories
SAMPLE_FILE = "/home/ciganche/Documents/AI_challange/data/dbb-participants.tsv"
TRAIN_DATA_DIR = "/home/ciganche/Documents/AI_challange/data/preprocessed/train/"
TEST_DATA_DIR = "/home/ciganche/Documents/AI_challange/data/preprocessed/test/"
GENERATED_DATA_DIR = "/home/ciganche/Documents/AI_challange/data/preprocessed/generated/"
SAMPLE_PREFIX = "sub-0"
EXTENSION = ".nii.gz"
SEGMENTATION_SUFFIX = "_segmented"

# output directories
STATS_OUTPUT = "/home/ciganche/Documents/AI_challange/data/nnUNet/Task001_BrainSeg/"
TRAIN_OUTPUT = "/home/ciganche/Documents/AI_challange/data/nnUNet/Task001_BrainSeg/imagesTr/"
LABEL_OUTPUT = "/home/ciganche/Documents/AI_challange/data/nnUNet/Task001_BrainSeg/labelsTr/"
TEST_OUTPUT = "/home/ciganche/Documents/AI_challange/data/nnUNet/Task001_BrainSeg/imagesTs/"
TEST_LABEL_OUTPUT = "/home/ciganche/Documents/AI_challange/data/nnUNet/Task001_BrainSeg/labelsTs/"

LEAVE_FOR_TEST = [940, 956, 935, 932]

pair_mat = pd.read_csv("data/generation_pairs.tsv", sep="\t", header=0).to_numpy()
generated_matrix = pair_mat[:, [0, 1, 2, 3, 4, 6, 7, 8, 9]]
ids = np.arange(0, len(generated_matrix))
generated_matrix = np.hstack((np.reshape(ids, (len(generated_matrix), 1)), generated_matrix))
# generated samples to exclude due to registration target being used in the test set
generated_matrix_to_keep = generated_matrix[np.logical_not(np.isin(generated_matrix[:, 1], LEAVE_FOR_TEST)), :]

# get list of triples (t1 file, segmentation file, id) for all the generated images being used
generated_pairs = []
for row in generated_matrix_to_keep:
    directory = GENERATED_DATA_DIR + "output_" + SAMPLE_PREFIX + str(row[6]) + "_2_" + SAMPLE_PREFIX + str(row[1])
    path_list = [f.path for f in os.scandir(directory)]
    if SEGMENTATION_SUFFIX in path_list[0]:
        pair = (path_list[1], path_list[0], row[0])
    else:
        pair = (path_list[0], path_list[1], row[0])
    generated_pairs.append(pair)
healthy_to_exclude = generated_matrix_to_keep[:, 6]

#%%
def get_train_data(dir, extension, sample_prefix, segmentation_suffix, healthy_to_exclude):
    ret_val = []    # a list of (image, segmentation) pairs
    id_list = []
    path_list = [f.path for f in os.scandir(dir)]
    for sample_path in path_list:
        if segmentation_suffix not in sample_path:
            sample_id = int(sample_path.split("/")[-1].replace(sample_prefix, "").replace(extension, ""))
            segmented_sample_path = sample_path.replace(extension, "") + segmentation_suffix + extension
            if segmented_sample_path in path_list and sample_id not in healthy_to_exclude:
                pair = (sample_path, segmented_sample_path)
                ret_val.append(pair)
                id_list.append(sample_id)
    return ret_val, id_list

def get_test_data(dir, extension, sample_prefix, segmentation_suffix, test_to_include):
    ret_val = []    # a list of (image, segmentation) pairs
    path_list = [f.path for f in os.scandir(dir)]
    for sample_path in path_list:
        if segmentation_suffix not in sample_path:
            sample_id = int(sample_path.split("/")[-1].replace(sample_prefix, "").replace(extension, ""))
            segmented_sample_path = sample_path.replace(extension, "") + segmentation_suffix + extension
            if segmented_sample_path in path_list and sample_id in test_to_include:
                pair = (sample_path, segmented_sample_path)
                ret_val.append(pair)
    return ret_val


# get list of pairs (t1 file, segmentation file) for all the train/test images being used
train_pairs, train_ids = get_train_data(TRAIN_DATA_DIR, EXTENSION, SAMPLE_PREFIX, SEGMENTATION_SUFFIX, healthy_to_exclude)
test_pairs = get_test_data(TEST_DATA_DIR, EXTENSION, SAMPLE_PREFIX, SEGMENTATION_SUFFIX, LEAVE_FOR_TEST)
#%% Prepare all

def process_pair(t1_seg_pair, train_output_dir, segmented_output_dir, sample_prefix, extension):
    t1_file = t1_seg_pair[0]
    segmented_file = t1_seg_pair[1]

    t1 = nib.load(t1_file)
    t1_array = t1.get_fdata()

    segmented = nib.load(segmented_file)
    segmented_array = segmented.get_fdata()
    segmented_resized = skimage.transform.resize(segmented_array, (256, 256, 256), mode="constant", order=0,
                                                 preserve_range=True)
    # final segmented array - also used as the mast
    segmented_final = np.round(segmented_resized, 0).astype(int)
    mask_array = (1 * (segmented_final > 0)).astype(int)
    mask_array = binary_dilation(mask_array, structure=None, iterations=1)

    t1_array = (t1_array / np.mean(t1_array)) * 100
    t1_resized = skimage.transform.resize(t1_array, (256, 256, 256), mode="constant", order=0, preserve_range=True)
    t1_final = (mask_array * t1_resized) + (2000 * (mask_array == 0))

    # output
    identifier = t1_file.split("/")[-1].replace(sample_prefix, "").replace(extension, "")

    t1_name = "sub" + "_" + identifier + "_" + "0000" + ".nii.gz"
    t1_output = nib.Nifti1Image(t1_final, affine=t1.affine, header=t1.header)
    nib.save(t1_output, train_output_dir + t1_name)

    seg_name = "sub" + "_" + identifier + ".nii.gz"
    if int(identifier) == 829:
        seg_output = nib.Nifti1Image(segmented_final, affine=t1.affine, header=t1.header)
    else:
        seg_output = nib.Nifti1Image(segmented_final, affine=segmented.affine, header=segmented.header)
    nib.save(seg_output, segmented_output_dir + seg_name)

def process_generated_pair(t1_seg_triple, train_output_dir, segmented_output_dir):
    t1_file = t1_seg_triple[0]
    segmented_file = t1_seg_triple[1]

    t1 = nib.load(t1_file)
    t1_array = t1.get_fdata()

    segmented = nib.load(segmented_file)
    segmented_array = segmented.get_fdata()
    segmented_resized = skimage.transform.resize(segmented_array, (256, 256, 256), mode="constant", order=0,
                                                 preserve_range=True)
    # final segmented array - also used as the mast
    segmented_final = np.round(segmented_resized, 0).astype(int)
    mask_array = (1 * (segmented_final > 0)).astype(int)
    mask_array = binary_dilation(mask_array, structure=None, iterations=1)

    t1_array = (t1_array / np.mean(t1_array)) * 100
    t1_resized = skimage.transform.resize(t1_array, (256, 256, 256), mode="constant", order=0, preserve_range=True)
    t1_final = (mask_array * t1_resized) + (2000 * (mask_array == 0))

    # output
    identifier = str("%03d" % t1_seg_triple[2])

    t1_name = "gen" + "_" + identifier + "_" + "0000" + ".nii.gz"
    t1_output = nib.Nifti1Image(t1_final, affine=t1.affine, header=t1.header)
    nib.save(t1_output, train_output_dir + t1_name)

    seg_name = "gen" + "_" + identifier + ".nii.gz"
    seg_output = nib.Nifti1Image(segmented_final, affine=segmented.affine, header=segmented.header)
    nib.save(seg_output, segmented_output_dir + seg_name)

#%%
# write all preprocessed files in nnUNet directory hierarchy under appropriate names
for pair in train_pairs:
    process_pair(pair, TRAIN_OUTPUT, LABEL_OUTPUT, SAMPLE_PREFIX, EXTENSION)
for triple in generated_pairs:
    process_generated_pair(triple, TRAIN_OUTPUT, LABEL_OUTPUT)
for pair in test_pairs:
    process_pair(pair, TEST_OUTPUT, TEST_LABEL_OUTPUT, SAMPLE_PREFIX, EXTENSION)

# write documents which samples are being used for what
all_samples = pd.read_csv(SAMPLE_FILE, sep="\t", header=0)
all_header = all_samples.columns
all_samples = all_samples.to_numpy()
# generated data
generated_header = ["id", "id_d", "age_bin_d", "age_d", "sex_d", "condition_d",
          "id_h", "age_bin_h", "age_h", "sex_h"]
generated_df = pd.DataFrame(generated_matrix_to_keep, columns=generated_header)
generated_df.to_csv(STATS_OUTPUT + "generated_used.tsv", sep="\t", header=True, index=False)
# train data
train_df = pd.DataFrame(all_samples[np.isin(all_samples[:, 0], train_ids), :], columns=all_header)
train_df.to_csv(STATS_OUTPUT + "train_used.tsv", sep="\t", header=True, index=False)
# test data
test_df = pd.DataFrame(all_samples[np.isin(all_samples[:, 0], LEAVE_FOR_TEST), :], columns=all_header)
test_df.to_csv(STATS_OUTPUT + "test_used.tsv", sep="\t", header=True, index=False)
