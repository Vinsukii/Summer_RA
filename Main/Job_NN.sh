#!/bin/sh
#
#$ -S /bin/sh
#
#$ -wd /vol/grid-solar/sgeusers/nguyenvinh1/Grid_RA/Jobs
#

# if [ -d /local/tmp/nguyenvinh1/$JOB_ID ]; then
#         cd /local/tmp/nguyenvinh1/$JOB_ID
# else
#         echo "Errors encountered! Run terminated..."
#         echo "LOCAL TMP location "
#         ls -la /local/tmp/nguyenvinh1
#         echo "Exiting..."
#         exit 1
# fi

seed=$SGE_TASK_ID
method=$1
size=$2
algo=$3

cd /vol/grid-solar/sgeusers/nguyenvinh1/Grid_RA/Main/Code
python3.9 test.py $seed $method $size $algo Y

echo "Run completed!"

#
# Email the result
#
#$ -M nguyenvinh1@myvuw.ac.nz
#$ -m e
#
