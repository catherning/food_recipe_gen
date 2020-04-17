
for i in "Seq2seq 04-14-04-45"  "Seq2seqAtt 04-14-08-38" "Seq2seqIngrAtt 04-14-18-23" "Seq2seqIngrPairingAtt 04-14-18-58" "Seq2seqTitlePairing 04-15-00-02" "Seq2seqCuisinePairing 04-16-06-10"
do 
    set -- $i
    
    python ./recipe_gen/main.py \
        --device ${1-0} \
        --batch-size 4 \
        --test False\
        --train-mode False \
        --max-ingr 10 \
        --max-length 100 \
        --sample-id a8ca14112c \
        --model-name $1 \
        --load-folder $2

done