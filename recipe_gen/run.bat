python ./recipe_gen/main.py ^
    --device 0 ^
    --resume ^
    --load-folder "03-09-15-13\train_model_03-09-20-24_6900.tar" ^
    --batch-size 4 ^
    --epoch 15 ^
    --n-iters 11500 ^
    --max-ingr 10 ^
    --max-length 100 ^
    --model-name Seq2seq ^
    --topk 20 
rem    --samples-max 1000