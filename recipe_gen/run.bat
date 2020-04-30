python -m cProfile -o "profilePairingAtt.txt"  ./recipe_gen/main.py ^
    --device 0 ^
    --batch-size 8 ^
    --epoch 20 ^
    --n-iters 6250 ^
    --max-ingr 10 ^
    --max-length 130 ^
    --model-name Seq2seq ^
    --num-gru-layers 2 ^
    --hidden-size 256
rem    --test False
rem    --resume ^
rem    --load-folder "03-09-15-13\train_model_03-09-20-24_6900.tar" ^