#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH

MODEL_IDs=("mistralai/Voxtral-Mini-3B-2507")

DATASETS_PATH="nithinraok/asr-leaderboard-datasets"

DATASETS=("fleurs_en" "fleurs_de" "fleurs_fr" "fleurs_it" "fleurs_es" "fleurs_pt"
        "mcv_en" "mcv_de" "mcv_es" "mcv_fr" "mcv_it"
        "mls_es" "mls_fr" "mls_it" "mls_pt")

num_datasets=${#DATASETS[@]}

BATCH_SIZE=48

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}

    for (( j=0; j<${num_datasets}; j++ ));
    do
        DATASET=${DATASETS[$j]}

        python run_eval.py \
            --model_id=${MODEL_ID} \
            --dataset_path=${DATASETS_PATH} \
            --dataset=${DATASET} \
            --split="test" \
            --device=0 \
            --batch_size=${BATCH_SIZE} \
            --max_eval_samples=-1

    done

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
