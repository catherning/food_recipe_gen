for file in recipe1m_test.pkl recipe1m_val.pkl
do
python ./evaluation/eval.py \
    --device ${1-0} \
    --batch-size 8 \
    --test \
    --load \
    --hidden-size 256 \
    --num-gru-layers 2 \
    --train-mode False \
    --max-ingr 10 \
    --max-length 20 \
    --model-name HierarchicalIngrPairingAtt \
    --eval-folder 06-04-22-36 \
    --unk-temp 0.1 \
    --test-file $file
done