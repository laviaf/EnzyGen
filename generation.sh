#!/bin/bash

data_path=data/enzyme_substrate_data_lucky_best.json

output_path=models
proteins=(2.4.1 3.2.2 2.7.10 4.6.1 2.1.1 3.6.4 3.1.4 3.6.5 1.14.13 3.4.21 3.5.1 3.4.19 3.5.2 2.4.2 4.2.1 1.1.1 1.2.1 2.7.11 2.3.1 3.1.3 2.7.1 2.7.4 3.1.1 2.5.1 2.7.7 2.6.1 4.1.1 1.11.1 3.6.1 1.14.14)

for element in ${proteins[@]}
do
generation_path=outputs/12_layers_rand_flip/${element}

mkdir -p ${generation_path}
mkdir -p ${generation_path}/pred_pdbs
mkdir -p ${generation_path}/tgt_pdbs

python3 fairseq_cli/validate.py ${data_path} \
--task geometric_protein_design \
--protein-task ${element} \
--dataset-impl-source "raw" \
--dataset-impl-target "coor" \
--path checkpoint_33_336000.pt \
--batch-size 1 \
--results-path ${generation_path} \
--skip-invalid-size-inputs-valid-test \
--valid-subset test \
--eval-aa-recovery
done