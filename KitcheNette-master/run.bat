
python main.py --save-prediction-unknowns True ^
                --model-name "smaller_kitchenette_trained.mdl" ^
                --unknown-path "./data/kitchenette_unknown_pairings.csv" ^
                --embed-path "./data/kitchenette_embeddings.pkl" ^
                --data-path "./data/kitchenette_dataset.pkl" ^
                --batch-size 64 ^
                --dist-fn "concat"