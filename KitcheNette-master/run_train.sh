CUDA_VISIBLE_DEVICES=${1-0} python3 ./KitcheNette-master/main.py \
                --model-name "kitchenette_pretrained.mdl" \
                --train true \
                --embed-path "./KitcheNette-master/data/kitchenette_embeddings.pkl" \
                --data-path "./KitcheNette-master/data/kitchenette_dataset.pkl" \
                --dist-fn "widedeep" \
                --checkpoint-dir "./KitcheNette-master/results/"
