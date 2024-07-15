mpic++ nccl.cc -o nccl -L/usr/local/cuda/lib64 -lcudart -I/data/bluefog/nccl-2.15.5/include -L/data/bluefog/nccl-2.15.5/lib -l nccl
mpirun -np 2 ./nccl
# -I/data/bluefog/nccl/include