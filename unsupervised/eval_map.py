for lg in es ar de fr zh; do
    SRC_EMB=/data2/jpzhang/sup_vecmap_static_out/$lg-en.$lg.our.mapped.txt
    TRG_EMB=/data2/jpzhang/sup_vecmap_static_out/$lg-en.en.our.mapped.txt
    TEST=/home/jpzhang/MUSE/our_pre_data/$lg-en.our.5000-6500.txt
    python3 eval_translation.py $SRC_EMB $TRG_EMB -d $TEST -t en -s $lg --tst test --retrieval csls --cuda
done