#!/bin/bash
#SBATCH -p gpu20,gpu16,gpu22
#SBATCH -t 05:00:00
#SBATCH --gres gpu:1
#SBATCH -o ./jobout.out
#SBATCH -e ./joberr.err

eval "$(conda shell.bash hook)"
echo "Conda hooked"

echo "Started Imagenet LT data creation"

# conda run -n GLTSSL --no-capture-output python imagenet_glt_generation.py --data_path /scratch/inf0/user/rnittala/ILSVRC/Data/CLS-LOC/train
conda run -n GLTSSL --no-capture-output python imagenet_glt_generation.py --data_path /BS/databases03/imagenet/ILSVRC2012/training_imgs
echo "Data creation done!!"

