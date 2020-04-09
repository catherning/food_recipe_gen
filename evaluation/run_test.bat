python ./evaluation/eval.py ^
        --device 0 ^
        --load-folder "04-08-11-34" ^
        --batch-size 4 ^
        --test ^
        --load ^
        --train-mode False ^
        --max-ingr 10 ^
        --max-length 100 ^
        --model-name Seq2seqAtt
