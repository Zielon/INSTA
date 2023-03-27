#!/bin/bash
COLOR='\033[0;32m'

echo -e "\n${COLOR}Installing dependencies..."

conda env create -f environment.yml

echo -e "\n${COLOR}RobustVideoMatting..."
mkdir -p dependencies
cd dependencies
git clone https://github.com/PeterL1n/RobustVideoMatting.git
cd RobustVideoMatting
cp ../../patches/matting .
git apply matting
rm -rf .git
mkdir model
wget https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth -O 'model/rvm_resnet50.pth' --no-check-certificate --continue

echo -e "\n${COLOR}face-parsing.PyTorch"
cd ../
git clone https://github.com/zllrunning/face-parsing.PyTorch.git
cd face-parsing.PyTorch
cp ../../patches/face-parsing .
git apply face-parsing
rm -rf .git

mkdir -p res
mkdir -p res/cp
wget https://keeper.mpdl.mpg.de/f/a3c400dc55b84b10a7d1/?dl=1 -O 'res/cp/79999_iter.pth' --no-check-certificate --continue
