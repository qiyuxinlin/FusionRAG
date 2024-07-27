#!/bin/bash
set -e  


echo "Installing python dependencies from requirements.txt"
pip install -r requirements-local_chat.txt

echo "Installing ktransformers cpuinfer"
mkdir -p ktransformers/ktransformers_ext/build
cd ktransformers/ktransformers_ext/build
cmake ..
cmake --build . --config Release

echo "Installing ktransformers gpu kernel, this may take about half an hour, please wait"
cd ../cuda
python setup.py install
cd ../../..
echo "Installation completed successfully"