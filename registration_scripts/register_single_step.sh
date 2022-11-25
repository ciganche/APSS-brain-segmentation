#!/bin/bash

# ANTs is considered to be in path - USE VERSION 2.4.2 since the previous ones don't have the save-state option.
# -j,-k parameters used to save and restore the previous iteration's state - parameter -a is required to use these.
# Since -a is required, transformation files are not saved as .mat/.nii.gz files but as .h5 binaries - all transformations are applied using those.
# Only the initial time step performs traslation, rigid, affine and SyN deformation (one SyN deformation step). All the others just one SyN deformation step.

dim=3

if [ $# -lt 5 ]; then
echo $( basename $0 ) "<fixed.nii.gz> <moving.nii.gz> <moving_segmented_nii.gz> <output_dir> <setting> [<n_threads>]"
echo ""
echo "Where <setting> {0, 1 or 2} determines the number of iterations on three different levels (1 for all conditions with lesser lesions - less iterations; 2 for complex lesions - more iterations; 0 to test the script - some unsufficient number of iterations)."
echo " and <n_threads> is optional and specifies the number of threads."
echo " ATNs version 2.4.2 must be in the PATH variable."
exit -1
fi

# - - - SETUP - - -

f=$1 ; m=$2 
m_segmented=$3
data_dir=$4
data_dir=`echo $data_dir | sed "s/\/$//"`
mysetting="$5"
n_threads=$6

f=${f}
m=${m}
m_segmented=${m_segmented}

[ -z ${n_threads} ] && { n_threads=2; }

ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=${n_threads}  # controls multi-threading
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS

echo "Number of threads used: "${n_threads}

if [[ ! -s $f ]] ; then echo no fixed $f ; exit; fi
if [[ ! -s $m ]] ; then echo no moving $m ;exit; fi
nm1=` basename $f | cut -d '.' -f 1 `
nm2=` basename $m | cut -d '.' -f 1 `

# - - - SET DEFORMATION PARAMETERS - - -

time_steps=5
if [[ $mysetting == "1" ]] ; then
	echo "- - - Fairly easy leasions mode (ACC, PFM, MDSC): 50x50x0 SyN iterations - - -"
  its=3x5x3
  percentage=0.3
  syn_step="50x50x0,-0.01,5"
  mysetting=forproduction
elif [[ $mysetting == "2" ]] ; then
	echo "- - - Complex lesions sample mode: 100x100x0 SyN iterations - - -"
  its=10000x111110x11110
  percentage=0.3
  syn_step="100x100x0,-0.01,5"
  mysetting=forproduction  
  
elif [[ $mysetting == "0" ]] ; then
  	echo "- - - Script testing mode: 5x5x3 SyN iterations - - -"
  its=3x5x5
  percentage=0.3
  syn_step="5x5x3,-0.01,5"
  mysetting=forproduction  
else
	echo "wrong setting"
	exit 1
fi

# - - - SET OUTPUT DIRECTORIES - - -

output_dir=${data_dir}/output_${nm2}_2_${nm1}
mkdir -p ${output_dir}

# - - - PERFORM INITIAL DEFORMATION - - -
nm=${output_dir}"/"${nm2}_2_${nm1}   # construct output prefix
antsRegistration -d $dim -r [ $f, $m ,1] \
                        -m mattes[  $f, $m , 1 , 32, regular, $percentage ] \
                         -t translation[ 0.1 ] \
                         -c [$its,1.e-8,20]  \
                        -s 4x2x1vox  \
                        -f 6x4x2 \
                        -m mattes[  $f, $m , 1 , 32, regular, $percentage ] \
                         -t rigid[ 0.1 ] \
                         -c [$its,1.e-8,20]  \
                        -s 4x2x1vox  \
                        -f 3x2x1  \
                        -m mattes[  $f, $m , 1 , 32, regular, $percentage ] \
                         -t affine[ 0.1 ] \
                         -c [$its,1.e-8,20]  \
                        -s 4x2x1vox  \
                        -f 3x2x1  \
                        -m mattes[  $f, $m , 0.5 , 32 ] \
                        -m cc[  $f, $m , 0.5 , 4 ] \
                         -t SyN[ .20, 3, 0 ] \
                         -c [ $syn_step ]  \
                        -s 1x0.5x0vox  \
                        -f 4x2x1 -u 1 -z 1 \
                       -o [${nm},${nm}_diff.nii.gz,${nm}_inv.nii.gz]  \
			-a 1
                
antsApplyTransforms -d $dim -i $m -r $f -n linear -t ${nm}Composite.h5 -o ${nm}_warped.nii.gz
antsApplyTransforms -d $dim -i $m_segmented -r $f -n MultiLabel -t ${nm}Composite.h5 -o ${nm}_segmented_warped.nii.gz
rm ${nm}Composite.h5
rm ${nm}InverseComposite.h5
rm ${nm}_diff.nii.gz
rm ${nm}_inv.nii.gz
