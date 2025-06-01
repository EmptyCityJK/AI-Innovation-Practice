#!/usr/bin/env bash
GPU_ID=0
data_dir=/root/datasets/OfficeHome
# # Office-Home
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain Art --tgt_domain Clipart | tee BNM_A2C.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain Art --tgt_domain RealWorld | tee BNM_A2R.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain Art --tgt_domain Product | tee BNM_A2P.log

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain Clipart --tgt_domain Art | tee BNM_C2A.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain Clipart --tgt_domain RealWorld | tee BNM_C2R.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain Clipart --tgt_domain Product | tee BNM_C2P.log

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain Product --tgt_domain Art | tee BNM_P2A.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain Product --tgt_domain RealWorld | tee BNM_P2R.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain Product --tgt_domain Clipart | tee BNM_P2C.log

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain RealWorld --tgt_domain Art | tee BNM_R2A.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain RealWorld --tgt_domain Product | tee BNM_R2P.log
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config BNM/BNM.yaml --data_dir $data_dir --src_domain RealWorld --tgt_domain Clipart | tee BNM_R2C.log