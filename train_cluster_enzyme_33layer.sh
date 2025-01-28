#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

data_path=data/enzyme_substrate_data_lucky_best_rm_all_mask.json

local_root=model_path
pretrained_model="esm2_t12_35M_UR50D"
output_path=${local_root}/enzygen_12layers_scale_coords

python3 fairseq_cli/train.py ${data_path} \
--profile \
--num-workers 0 \
--distributed-world-size 1 \
--save-dir ${output_path} \
--task geometric_protein_design \
--dataset-impl-source "raw" \
--dataset-impl-target "coor" \
--criterion geometric_protein_loss --encoder-factor 1.0 --decoder-factor 1.0 \
--arch geometric_protein_model_esm \
--encoder-embed-dim 480 \
--egnn-mode "rm-node" \
--tanh False \
--coordinate-scaling 0.1 \
--decoder-layers 3 \
--pretrained-esm-model ${pretrained_model} \
--knn 30 \
--dropout 0.3 \
--optimizer adam --adam-betas '(0.9,0.98)' \
--lr 3e-4 --lr-scheduler inverse_sqrt \
--stop-min-lr '1e-10' --warmup-updates 4000 \
--warmup-init-lr '5e-5' \
--clip-norm 10 \
--ddp-backend legacy_ddp \
--log-format 'simple' --log-interval 10 \
--max-tokens 1024 \
--update-freq 1 \
--max-update 1000000 \
--max-epoch 100 \
--validate-after-updates 3000 \
--validate-interval-updates 3000 \
--save-interval-updates 3000 \
--valid-subset valid \
--max-sentences-valid 8 \
--validate-interval 1 \
--save-interval 1 \
--keep-interval-updates	10 \
--skip-invalid-size-inputs-valid-test

