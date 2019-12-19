python3 ./main.py \
                --save-prediction-unknowns True \
                --model-name "kitchenette_pretrained.mdl" \
                --unknown-path "./data/main_pairing_prediction.csv" \
                --embed-path "./data/kitchenette_embeddings.pkl" \
                --data-path "./data/kitchenette_dataset.pkl" \
                --dist-fn "widedeep"
