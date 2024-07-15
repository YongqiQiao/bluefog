pip uninstall bluefog -y
make clean
BLUEFOG_WITH_NCCL=1 pip install -e . -vv