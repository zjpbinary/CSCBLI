set -e

static_vecmap_dir=
context_vecmap_dir=


for tgt in de fr; do
src=en
model_path=
echo $src $tgt
CUDA_VISIBLE_DEVICES=2 python test.py  --model_path $model_path \
        --dict_path test_dict/$src-$tgt.test.txt  --mode v1 \
        --src_lang $src --tgt_lang $tgt --lambda_w1 0.05

done
for src in de fr; do
tgt=en
model_path=
echo $src $tgt
CUDA_VISIBLE_DEVICES=2 python test.py  --model_path $model_path \
        --dict_path test_dict/$src-$tgt.test.txt  --mode v1 \
        --src_lang $src --tgt_lang $tgt --lambda_w1 0.05

done

