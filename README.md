# CSCBLI
Code for our ACL Findings 2021 paper, "Combining Static Word Embedding and Contextual Representations for Bilingual Lexicon Induction".  

## Requirements
python >= 3.6  
numpy >= 1.9.0  
pytorch >= 1.0  
## Supervised
### How to train

```
CUDA_VISIBLE_DEVICES=0 python train.py --src_lang $lg --tgt_lang en\
        --static_src_emb_path $ssemb --static_tgt_emb_path $stemb\
        --context_src_emb_path $csemb --context_tgt_emb_path $ctemb\
        --train_data_path $data_path --save_path $save_path
```

```
--static_src_emb_path   aligned source static embedding path 
--static_tgt_emb_path   aligned target static embedding path
--context_src_emb_path  source context embedding path
--context_tgt_emb_path  target context embedding path
```

### How to Test

```
CUDA_VISIBLE_DEVICES=0 python test_on_all_word.py --src_lang $lg\
        --tgt_lang en --model_path $model_path\
        --dict_path $dict_path\
        --vecmap_context_src_emb_path $vcpath\
        --vecmap_context_tgt_emb_path $vspath\
        --vecmap
```

```
--vecmap_context_src_emb_path aligned source context embedding path
--vecmap_context_tgt_emb_path aligned target context embedding path
--vecmap use interpolation method, else unified method
```

## Unsupervised
### How to train

```
lg=ar
CUDA_VISIBLE_DEVICES=0 python train.py --src_lang en --tgt_lang $lg --mode $md\
  --static_src_emb_path $ssemb --static_tgt_emb_path $stemb\
  --context_src_emb_path $csemb --context_tgt_emb_path $ctemb\
   --save_path $save_path 
```

```
--static_src_emb_path   aligned source static embedding path 
--static_tgt_emb_path   aligned target static embedding path
--context_src_emb_path  source context embedding path
--context_tgt_emb_path  target context embedding path
```

### How to Test

```
src=ar
tgt=en
model_path=../checkpoints/$src-$tgt-add_orign_nw.pkl_last
CUDA_VISIBLE_DEVICES=0 python test.py  --model_path $model_path \
        --dict_path ../$src-$tgt.5000-6500.txt  --mode v2 \
        --src_lang $src --tgt_lang $tgt  \
        --reload_src_ctx   $path1 \
        --reload_tgt_ctx   $path2 --lambda_w1 0.11
```

```
--mode type    use v1 for unified method and v2 for interpolated 
--lambda_w1    the weight for interpolation
--reload_src_ctx   aligned source context embedding
--reload_tgt_ctx   aligned targte context embedding
```


