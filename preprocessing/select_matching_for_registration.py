import pandas as pd
import numpy as np
import nibabel as nib
import os

SAMPLE_FILE = "/home/ciganche/Documents/AI_challange/data/dbb-participants.tsv"
TRAIN_DATA_DIR = "/home/ciganche/Documents/AI_challange/data/preprocessed/train/"
TEST_DATA_DIR = "/home/ciganche/Documents/AI_challange/data/preprocessed/test/"
SAMPLE_PREFIX = "sub-0"
SAMPLE_SUFFIX = ".nii.gz"
SEGMENTATION_SUFFIX = "_segmented"

GENERATE_FOR_EACH = 3

def get_nii(directory, prefix, extension):
    sample_list = []
    id_list = []
    for f in os.scandir(directory):
        if prefix in f.path and extension in f.path:
            if SEGMENTATION_SUFFIX in f.path:
                continue
            val = f.path.split("/")[-1]
            sample_list.append(f.path)
            id_list.append(int(val.replace(prefix, "").replace(extension, "")))
    return sample_list, id_list


train_sample_names, train_ids = get_nii(TRAIN_DATA_DIR, SAMPLE_PREFIX, SAMPLE_SUFFIX)
test_sample_names, test_ids = get_nii(TEST_DATA_DIR, SAMPLE_PREFIX, SAMPLE_SUFFIX)
sample_matrix = pd.read_csv(SAMPLE_FILE, header=0, sep="\t").to_numpy()

segmentation_file_names = []
for t in train_sample_names:
    segmentation_file_names.append(t.replace(SAMPLE_SUFFIX, SEGMENTATION_SUFFIX+SAMPLE_SUFFIX))

#%% select the 207 train samples and the 37 test
def get_by_id(matrix, ids):
    to_keep = []
    for id in ids:
        for i in range(0, len(matrix)):
            if matrix[i, 0] == id:
                to_keep.append(i)

    ret_val = matrix[to_keep, :]
    return ret_val[:, [0, 1, 2, 3, 5]]


train_matrix = get_by_id(sample_matrix, train_ids)
train_matrix = np.hstack((train_matrix, np.reshape(train_sample_names, (len(train_sample_names), 1))))
# add also segmentation file paths to train_matrix
train_matrix = np.hstack((train_matrix, np.reshape(segmentation_file_names, (len(segmentation_file_names), 1))))

test_matrix = get_by_id(sample_matrix, test_ids)
test_matrix = np.hstack((test_matrix, np.reshape(test_sample_names, (len(test_sample_names), 1))))
#%% select only the ones of size (182, 218, 182) because all the test stuff is as such
def get_by_shape(matrix):
    shapes_vector = []
    for sample in matrix:
        t1_nii = nib.load(sample[5])
        shapes_vector.append(str(t1_nii.shape))
    to_keep = np.where(np.asarray(shapes_vector) == "(182, 218, 182)")[0]
    return matrix[to_keep, :]


train_matrix = get_by_shape(train_matrix)
test_matrix = get_by_shape(test_matrix)

#%% Put HD at the first place to prioritize them
hd_indices = np.where(test_matrix[:, 4] == "HD")[0]
other = np.setdiff1d(np.arange(0, len(test_matrix)), hd_indices)
test_matrix = test_matrix[np.append(hd_indices, other), :]

#%% Categorize age:
bins = [0, 0.5, 1, 2, 5, 8, 11, 12, 17, 22]
train_age_category = np.digitize(train_matrix[:, 2], bins, False)  # False to get [a,b) intervals
train_matrix[:, 1] = train_age_category

test_age_category = np.digitize(test_matrix[:, 2], bins, False)  # False to get [a,b) intervals
test_matrix[:, 1] = test_age_category


#%% Match samples - no repetition - find the same age bin ones with minimal age difference:
# 0: ID
# 1: age bin
# 2: age
# 3: sex
# 4: state
# 5: location of nii.gz file
# (6): segmentation location

def get_minimal_age_diff(sample, train_mat, category_diff, already_chosen):
    potential_indices = np.where(train_mat[:, 1] == sample[1] + category_diff)[0]
    selected = None
    mmmin = 1500
    for p in potential_indices:
        if p not in already_chosen:
            if abs(train_mat[p, 2] - sample[2]) < mmmin:
                selected = p
                mmmin = abs(train_mat[p, 2] - sample[2])
    return selected

chosen_index = []
ret_val = []
for i in range(0, GENERATE_FOR_EACH):
    for malformed_sample in test_matrix:
            malformed_sample[0] = int(malformed_sample[0])
            # select from same age group
            selected_index = get_minimal_age_diff(malformed_sample, train_matrix, 0, chosen_index)

            # higher age category
            if selected_index == None:
                selected_index = get_minimal_age_diff(malformed_sample, train_matrix, 1, chosen_index)

            # lower age category
            if selected_index == None:
                selected_index = get_minimal_age_diff(malformed_sample, train_matrix, -1, chosen_index)

            # higher +2 age category
            if selected_index == None:
                selected_index = get_minimal_age_diff(malformed_sample, train_matrix, 2, chosen_index)

            # lower -2 age category
            if selected_index == None:
                selected_index = get_minimal_age_diff(malformed_sample, train_matrix, -2, chosen_index)

            # higher +3 age category
            if selected_index == None:
                selected_index = get_minimal_age_diff(malformed_sample, train_matrix, 3, chosen_index)

            # lower -3 age category
            if selected_index == None:
                selected_index = get_minimal_age_diff(malformed_sample, train_matrix, -3, chosen_index)

            if selected_index == None:
                print("No matching ones for: " + str(malformed_sample[0]))
            else:
                chosen_index.append(selected_index)
                ret_val.append(np.append(malformed_sample, train_matrix[selected_index, :]))

#%% sort by malformed id, write output to tsv

ret_val_matrix = np.asarray(ret_val)
sorted_indices = np.argsort(ret_val_matrix[:, 0])
ret_val_matrix = ret_val_matrix[sorted_indices, :]

header = ["id_d", "age_bin_d", "age_d", "sex_d", "condition_d", "volume_d",
          "id_h", "age_bin_h", "age_h", "sex_h", "condition_h", "volume_h", "segmentation_h"]
output_df = pd.DataFrame(ret_val_matrix, columns=header)
output_df.to_csv("data/generation_pairs.tsv", header=True, index=False, sep="\t")

