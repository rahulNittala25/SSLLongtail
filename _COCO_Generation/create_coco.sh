#!/bin/bash
#SBATCH -p cpu20
#SBATCH -t 05:00:00
#SBATCH -o ./jobout_3.out
#SBATCH -e ./joberr_3.err

eval "$(conda shell.bash hook)"
echo "Conda hooked"

echo "Started COCO LT data creation"

# conda run -n GLTSSL --no-capture-output python mscoco_glt_generation.py --data_path /BS/SSLLongTail/nobackup/COCO_data --anno_path /BS/SSLLongTail/nobackup/annotations/ --attribute_path ./cocottributes_py3.jbl

#echo "MSCOCO GLT GENERATION DONE!"

conda run -n GLTSSL --no-capture python mscoco_glt_crop.py --data_path /BS/SSLLongTail/nobackup/COCO_data/ --output_path /BS/SSLLongTail/nobackup/COCO_BL_balanced

echo "Finished Cropping!!"

