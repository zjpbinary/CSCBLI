set -e

#static_vecmap_dir=/data4/bjji/source/No_sup_vecmap/vecmap_out
#context_dir=/data4/bjji/source/No_sup_vecmap/our_real_avg_bpe_vec

# for lg in es ar de zh fr; do
#     for md in add_orign_nw; do
#         ssemb=$static_vecmap_dir/en-$lg.en.our.mapped.txt
#         stemb=$static_vecmap_dir/en-$lg.$lg.our.mapped.txt
#         csemb=$context_dir/en.avg.vec
#         ctemb=$context_dir/$lg.avg.vec
#         # data_path=/home/jpzhang/py36pt12/vecmap/pre_train_data/en-$lg.train.csls.data
#         save_path=./checkpoints_dynamic_neg/en-$lg-$md.pkl
#         CUDA_VISIBLE_DEVICES=1 python train.py --src_lang en --tgt_lang $lg --mode $md\
#         --static_src_emb_path $ssemb --static_tgt_emb_path $stemb\
#         --context_src_emb_path $csemb --context_tgt_emb_path $ctemb\
#          --save_path $save_path --vecmap_context_src_emb_path /data4/bjji/vecmap_avg_out/en-$lg.en.our.mapped.txt \
#          --vecmap_context_tgt_emb_path /data4/bjji/vecmap_avg_out/en-$lg.$lg.our.mapped.txt
#     done
# done



static_vecmap_dir=/data2/jpzhang/Wacky/unsup_align_vec
context_dir=/data2/jpzhang/Wacky
#vecmap_context_dir=/data2/jpzhang/new_unsup_vecmap_avg_out
for lg in de fr ; do
    for md in add_orign_nw; do
        ssemb=$static_vecmap_dir/$lg-en.$lg.our.mapped.txt
        stemb=$static_vecmap_dir/$lg-en.en.our.mapped.txt
        csemb=$context_dir/$lg.vec
        ctemb=$context_dir/en.vec
        #vsemb=$context_dir/en-$lg.en.4000.our.mapped.txt
        #vtemb=$context_dir/en-$lg.$lg.4000.our.mapped.txt
        save_path=/data2/jpzhang/Wacky/bucc_unsup_checkpoints/$lg-en-$md.pkl
        CUDA_VISIBLE_DEVICES=2 python train.py --src_lang $lg --tgt_lang en --mode $md\
        --static_src_emb_path $ssemb --static_tgt_emb_path $stemb \
        --context_src_emb_path $csemb --context_tgt_emb_path $ctemb \
        --save_path $save_path 
        #--vecmap_context_src_emb_path $vsemb
        #--vecmap_context_tgt_emb_path $vtemb
    done
done
