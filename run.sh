#!/bin/sh
/usr/local/cuda/bin/nvcc src/kernels.cu `pkg-config opencv --cflags --libs` main.cpp src/IP.cpp -o application
#./application