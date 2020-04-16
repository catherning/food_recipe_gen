for file in recipe1m_val.pkl recipe1m_test.pkl
do
python ./evaluation/eval.py \
    --device ${1-0} \
    --batch-size 4 \
    --test \
    --load \
    --train-mode False \
    --max-ingr 10 \
    --max-length 100 \
    --model-name Seq2seqTitlePairing \
    --load-folder "04-15-00-02" \
    --test-file $file
done

#recipe1m_test.pkl