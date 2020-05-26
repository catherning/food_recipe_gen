python ./recipe_gen/main.py \
    --device ${1-0} \
    --batch-size 8 \
    --epoch 50 \
    --n-iters 11500 \
    --max-ingr 10 \
    --max-length 20 \
    --model-name HierarchicalTitlePairing \
    --hidden-size 256 \
    --num-gru-layers 2 \
    --resume \
    --load-folder 05-14-20-31
    #--train-file recipe1m_train_main.pkl \
    #--test-file recipe1m_test_main.pkl 
    # Seq2seq 05-11-18-42_m
    # Seq2seqIngrAtt 05-08-23-07_m
    # HierarchicalSeq2seq 05-16-04-54_m 05-18-22-52
    # HierarchicalIngrAtt 05-20-06-16
    # HierarchicalCuisinePairing 05-14-20-29_m 05-20-06-20 05-21-18-10
    # HierarchicalIngrPairingAtt 05-16-22-08 05-21-18-50
    # HierarchicalTitlePairing 05-14-20-31