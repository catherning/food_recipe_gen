
for i in "Seq2seq 04-14-04-45"  "Seq2seqAtt 04-14-08-38" "Seq2seqIngrAtt 04-14-18-23" "Seq2seqIngrPairingAtt 04-14-18-58" "Seq2seqTitlePairing 04-15-00-02" 
do 
    set -- $i
    
    python ./recipe_gen/main.py \
        --device 0 \
        --batch-size 4 \
        --test False \
        --train-mode False \
        --load \
        --max-ingr 10 \
        --max-length 100 \
        --sample-id a8ca14112c \
        --model-name $1 \
        --load-folder $2 \
         --test-file recipe1m_val.pkl

done

    python ./recipe_gen/main.py \
        --device 0 \
        --batch-size 4 \
        --test False \
        --train-mode False \
        --load \
        --max-ingr 10 \
        --max-length 100 \
        --sample-id a8ca14112c \
        --model-name Seq2seqCuisinePairing \
        --load-folder 04-16-06-10 \
        --train-file recipe1m_train_cuisine_nn.pkl \
        --test-file recipe1m_val_cuisine_nn.pkl
