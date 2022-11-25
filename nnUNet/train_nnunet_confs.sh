# set device
export CUDA_VISIBLE_DEVICES=1


# new trainer classes nnUNetTrainerMoj and nnUNetTrainerMojCascade have been created to extend the classes nnUNetTrainerV2 and nnUNetTrainerV2Cascade respectively
# instead of training for 1000 epoch, all configurations are trained for 75

# train all configurations
for fold in $(seq 0 4)
do
	nnUNet_train 2d nnUNetTrainerMoj Task001_BrainSeg $fold --npz
done

echo "* * * 3D fullres * * *"

for fold in $(seq 0 4)
do
        nnUNet_train 3d_fullres nnUNetTrainerMoj Task001_BrainSeg $fold --npz
done

echo "* * * 3D lowres * * *"
for fold in $(seq 0 4)
do
        nnUNet_train 3d_lowres nnUNetTrainerMoj Task001_BrainSeg $fold --npz
done

echo "* * * CASCADE * * *"
for fold in $(seq 0 4)
do
        nnUNet_train 3d_cascade_fullres nnUNetTrainerMojCascade Task001_BrainSeg $fold --npz
done


# find best configuration of all trained
nnUNet_find_best_configuration -m 2d 3d_fullres 3d_lowres 3d_cascade_fullres -t 001

# use the best identified configuration for predicting: ensemble of 2D U-Net and full resolution 3D U-Net

nnUNet_predict -i /home/hit/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task001_BrainSeg/imagesTs -o /home/hit/nnUNet/results/output_models/e1 -tr nnUNetTrainerMoj -ctr nnUNetTrainerV2CascadeFullRes -m 2d -p nnUNetPlansv2.1 -t Task001_BrainSeg -z

nnUNet_predict -i /home/hit/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task001_BrainSeg/imagesTs -o /home/hit/nnUNet/results/output_models/e2 -tr nnUNetTrainerMoj -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 -t Task001_BrainSeg -z
nnUNet_ensemble -f /home/hit/nnUNet/results/output_models/e1 /home/hit/nnUNet/results/output_models/e2 -o /home/hit/nnUNet/results/output_models/ensemble -pp /home/hit/nnUNet/results/nnUNet/ensembles/Task001_BrainSeg/ensemble_2d__nnUNetTrainerMoj__nnUNetPlansv2.1--3d_fullres__nnUNetTrainerMoj__nnUNetPlansv2.1/postprocessing.json


