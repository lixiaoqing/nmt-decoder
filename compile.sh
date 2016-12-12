set -x
nvcc -arch=sm_37 -O3 -lcublas -lcurand -o a decoder.cu
