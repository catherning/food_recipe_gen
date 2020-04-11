FOR %%m IN (Seq2seqIngrAtt) DO (
    python -m cProfile -o ./recipe_gen/profiles/prof%%mTopK.txt  ./recipe_gen/main.py ^
        --device 0 ^
        --batch-size 4 ^
        --epoch 1 ^
        --n-iters 1000 ^
        --max-ingr 10 ^
        --max-length 100 ^
        --model-name %%m ^
        --test False ^
        --nucleus-sampling False ^
        --topk 10
) 

rem Seq2seq Seq2seqAtt Seq2seqIngrAtt Seq2seqIngrPairingAtt