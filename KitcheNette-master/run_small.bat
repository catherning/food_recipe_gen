
python "D:\Documents\THU\food_recipe_gen\KitcheNette-master\main.py" ^
                --model-name smaller_kitchenette_trained.mdl  ^
                --data-path ./data/kitchenette_dataset.pkl  ^
                --batch-size 64  ^
                --dist-fn concat