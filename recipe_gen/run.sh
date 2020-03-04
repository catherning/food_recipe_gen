python ./recipe_gen/main.py \
    --device ${1-0} \
    --batch-size 4 \
    --epoch 20 \
    --n-iters 11500 \
    --max-ingr 10 \
    --max-length 15 \
    --model-name HierarchicalSeq2seq \
    --topk 6