
src=en
tgt=es

model_path=
CUDA_VISIBLE_DEVICES=4 python unsupervised_tuning.py --model_path $model_path \
        --src_lang $src --tgt_lang $tgt \
        --reload_src_ctx  /$src-$tgt.$src.our.mapped.txt \
        --reload_tgt_ctx /$src-$tgt.$tgt.our.mapped.txt