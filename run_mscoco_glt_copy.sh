#!/bin/bash
#SBATCH -p gpu20,gpu16,gpu22
#SBATCH -t 12:00:00
#SBATCH --gres gpu:2
#SBATCH -o ./jobout.out
#SBATCH -e ./joberr.err

eval "$(conda shell.bash hook)"
echo "Conda hooked"

echo "Started training fixmatch on MSCOCO GLT"

#  conda run -n GLTSSL --no-capture-output python imagenet_glt_generation.py --data_path /scratch/inf0/user/rnittala/ILSVRC/Data/CLS-LOC/train
conda run -n GLTSSL --no-capture-output python train_mscoco_glt.py --img_size 112 --epoch 500 --val-iteration 500 --out ./results/mscoco_glt_112x112/fixmatch/baseline/resnet50_tau50 --gpu 0,1 --batch-size 64 --tau 0.85
echo "Fixmatch Done!!"
