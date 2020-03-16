CUDA_VISIBLE_DEVICES=${1-0} python3 ./KitcheNette_master/main.py \
                --model-name "kitchenette_pretrained.mdl" \
                --train true \
                --epoch 50 \
                --embed-path "./KitcheNette_master/data/kitchenette_embeddings.pkl" \
                --data-path "./KitcheNette_master/data/kitchenette_dataset.pkl" \
                --dist-fn "widedeep" \
                --checkpoint-dir "./KitcheNette_master/results/"
