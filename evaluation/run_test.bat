python ./evaluation/eval.py ^
        --device 0 ^
        --eval-folder "04-27-22-16" ^
        --batch-size 4 ^
        --test ^
        --load ^
        --train-mode False ^
        --max-ingr 10 ^
        --max-length 100 ^
        --model-name HierarchicalSeq2seq
        REM --test-file "recipe1m_val_cuisine_nn.pkl" ^ 
        REM --train-file "recipe1m_train_nn_main.pkl"
