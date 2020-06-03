python ./recipe_gen/main.py \
    --device ${1-0} \
    --batch-size 8 \
    --epoch 40 \
    --n-iters 11500 \
    --max-ingr 10 \
    --max-length 100 \
    --model-name Seq2seqTitlePairing \
    --hidden-size 256 \
    --num-gru-layers 2 \
    --unk-temp 0 \
    --resume \
    --load-folder 05-11-18-55_m
    #--train-file recipe1m_train_main.pkl \
    #--test-file recipe1m_test_main.pkl 

    # Seq2seq 40 05-18-22-47
    # Seq2seqIngrAtt 40 05-18-22-49
    # Seq2seqIngrPairingAtt 30 05-08-23-05_m 40 06-03-22-07
    # Seq2seqTitlePairing 30 05-11-18-55_m
    # HierarchicalSeq2seq 05-18-22-52
    # HierarchicalIngrAtt 40 05-20-06-16
    # HierarchicalCuisinePairing 05-21-18-10
    # HierarchicalIngrPairingAtt 50 05-23-19-06
    # HierarchicalTitlePairing 05-23-19-07