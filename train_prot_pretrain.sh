#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

data_path=data/processed_pdb/metadata_filtered.csv

local_root=model_path
pretrained_model="esm2_t33_650M_UR50D"
output_path=${local_root}/model_w_rot_avg_loss

python3 fairseq_cli/train.py ${data_path} \
--profile \
--num-workers 0 \
--distributed-world-size 1 \
--save-dir ${output_path} \
--task geometric_protein_design_pdb \
--dataset-impl-source "raw" \
--dataset-impl-target "coor" \
--criterion geometric_protein_pdb_loss --aa-type-factor 1.0 --trans-factor 1.0 --rot-factor 1.0 \
--arch geometric_protein_model_pdb_esm \
--encoder-embed-dim 1280 \
--egnn-mode "pdb-pretrain" \
--decoder-layers 3 \
--inter-layers 11 \
--pretrained-esm-model ${pretrained_model} \
--knn 30 \
--dropout 0.3 \
--optimizer adam --adam-betas '(0.9,0.98)' \
--lr 3e-4 --lr-scheduler inverse_sqrt \
--stop-min-lr '1e-10' --warmup-updates 4000 \
--warmup-init-lr '5e-5' \
--clip-norm 0.0001 \
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

