python ./evaluation/eval.py \
    --device ${1-0} \
    --load-folder "04-07-17-53" \
    --batch-size 4 \
    --test \
    --load \
    --train-mode False \
    --max-ingr 10 \
    --max-length 100 \
    --model-name Seq2seq