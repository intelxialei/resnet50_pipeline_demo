#! /bin/bash
#source ~/intel/oneapi/setvars.sh 
#oneapi_version=latest
oneapi_version=2022.2.1
tbb_version=2021.7.1
export CUDA_VISIBLE_DEVICES=0
export ONEAPI_HOME=/mnt/nfs/home/lxia1/intel/oneapi # OneAPI default installed path
source ${ONEAPI_HOME}/tbb/${tbb_version}/env/vars.sh
source ${ONEAPI_HOME}/mkl/${oneapi_version}/env/vars.sh
export LD_PRELOAD=$ONEAPI_HOME/mkl/${oneapi_version}/lib/intel64/libmkl_core.so:$ONEAPI_HOME/mkl/${oneapi_version}/lib/intel64/libmkl_tbb_thread.so

#export DPCPP_HOME=${ONEAPI_HOME}/compiler/latest/linux
export DPCPP_HOME=${ONEAPI_HOME}/compiler/${oneapi_version}/linux
export LD_LIBRARY_PATH=${DPCPP_HOME}/lib:${DPCPP_HOME}/compiler/lib/intel64_lin:${LD_LIBRARY_PATH}


#export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
#export ITEX_ONEDNN_GRAPH=True
# export TF_CPP_MAX_VLOG_LEVEL=1
export TF_NUM_INTEROP_THREADS=1 

export ITEX_SET_BACKEND=GPU
export ITEX_LAYOUT_OPT=0

python3 benchmark_inference.py --batch_size 128 --data_dir /home/lxia1/data/imagenet2012/val/ --version=1 --no_xla --pb_path "/home/lxia1/Resnet50v2/resnet50_v1.pb" --preprocess_method=3 --async_inference 
