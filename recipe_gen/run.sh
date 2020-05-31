python ./recipe_gen/main.py \
    --device ${1-0} \
    --batch-size 8 \
    --epoch 50 \
    --n-iters 11500 \
    --max-ingr 10 \
    --max-length 20 \
    --model-name HierarchicalIngrAtt \
    --hidden-size 256 \
    --num-gru-layers 2 \
    --unk-temp 0 \
    --resume \
    --load-folder 05-20-06-16
    #--train-file recipe1m_train_main.pkl \
    #--test-file recipe1m_test_main.pkl 

    # Seq2seq 05-11-18-42_m
    # Seq2seqIngrAtt 05-08-23-07_m
    # HierarchicalSeq2seq 05-18-22-52
    # HierarchicalIngrAtt 40 05-20-06-16
    # HierarchicalCuisinePairing 05-21-18-10
    # HierarchicalIngrPairingAtt 50 05-23-19-06
    # HierarchicalTitlePairing 05-23-19-07