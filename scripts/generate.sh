#!/bin/bash

USER="$(whoami)"

PYTHON_ENV=/home/${USER}/miniconda3/etc/profile.d/conda.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source ${PYTHON_ENV}

INPUT="empty"
OUTPUT="empty"
TEST_FRAMES=350

if [ -n "$1" ]; then INPUT=${1}; fi
if [ -n "$2" ]; then OUTPUT=${2}; fi
if [ -n "$3" ]; then TEST_FRAMES=${3}; fi

conda activate rta

echo "Generate insta transform file"
python transforms.py -i ${INPUT} -o ${OUTPUT} -t ${TEST_FRAMES}

echo "Eyes simplification"
python simplification.py -i ${OUTPUT}

echo "Robust Video Matting"
cd dependencies/RobustVideoMatting
python inference.py --variant resnet50 --checkpoint model/rvm_resnet50.pth --device cuda:0 --input-source ${OUTPUT}/background/ --output-type png_sequence --output-composition ${OUTPUT}/matted/ --num-workers 8
cd ../

echo "Face Parsing"
cd face-parsing.PyTorch
python test.py --actor ${OUTPUT}

echo "Postprocessing"
cd ../../
python postprocess.py -i ${OUTPUT}

rm -rf ${OUTPUT}/background/

echo "End!"

# ./generate.sh /home/wojciech/projects/metrical-tracker/output/duda/ /home/wojciech/projects/INSTA/data/duda/ 10
