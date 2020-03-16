CUDA_VISIBLE_DEVICES=${1-0} python3 ./KitcheNette_master/main.py \
                --save-prediction-unknowns True \
                --model-name "kitchenette_pretrained.mdl" \
                --unknown-path "./KitcheNette_master/data/main_pairing_prediction.csv" \
                --embed-path "./KitcheNette_master/data/kitchenette_embeddings.pkl" \
                --data-path "./KitcheNette_master/data/kitchenette_dataset.pkl" \
                --dist-fn "widedeep" \
                --checkpoint-dir "./KitcheNette_master/results/"
