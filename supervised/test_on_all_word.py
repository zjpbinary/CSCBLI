import torch
from torch.utils.data import Dataset, DataLoader
import io
import numpy as np
import torch.nn as nn
import argparse
import pdb
from dictionary import Dictionary
from model import Discriminator
from preprocess import MyDataset


parser = argparse.ArgumentParser(description='Discriminator testing')
parser.add_argument("--model_path", type=str, default='model.pkl', help="point model path")
parser.add_argument("--dict_path", type=str, default='', help="point test dic")
parser.add_argument("--src_lang", type=str, default='en', help="point src lang")
parser.add_argument("--tgt_lang", type=str, default='ar', help="point tgt lang")
parser.add_argument("--vecmap_context_src_emb_path", type=str, default='', help="point vecmap_c_src_emb_path")
parser.add_argument("--vecmap_context_tgt_emb_path", type=str, default='', help="point vecmap_c_tgt_emb_path")
parser.add_argument("--static_src_emb_path", type=str, default='', help="src_emb_path")
parser.add_argument("--static_tgt_emb_path", type=str, default='', help="tgt_emb_path")
parser.add_argument("--context_src_emb_path", type=str, default='', help="src_emb_path")
parser.add_argument("--context_tgt_emb_path", type=str, default='', help="tgt_emb_path")
parser.add_argument("--orign", action='store_true', help="tgt_emb_path")
parser.add_argument("--vecmap", action='store_true', help="tgt_emb_path")
params = parser.parse_args()


def test(model_path, dict_path):

    model = torch.load(model_path)
    
    
    if params.vecmap_context_src_emb_path != '' and params.vecmap_context_tgt_emb_path != '':
        print('reload emb')
        model.reload_vecmap_context_emb(params.vecmap_context_src_emb_path, params.vecmap_context_tgt_emb_path)
        print('reload successfully')
    
    if params.static_src_emb_path != '' and params.static_tgt_emb_path != '':
        print('reload emb')
        model.reload_static_emb(params.static_src_emb_path, params.static_tgt_emb_path)
        print('reload successfully')
    if params.context_src_emb_path != '' and params.context_tgt_emb_path != '':
        print('reload emb')
        model.reload_context_emb(params.context_src_emb_path, params.context_tgt_emb_path)
        print('reload successfully')
    
    s_src_w2i = model.static_src_dico.word2id
    s_tgt_w2i = model.static_tgt_dico.word2id
    c_src_w2i = model.context_src_dico.word2id
    c_tgt_w2i = model.context_tgt_dico.word2id
    if params.vecmap_context_src_emb_path != '' and params.vecmap_context_tgt_emb_path != '':
        vc_src_w2i = model.vecmap_context_src_dico.word2id
        vc_tgt_w2i = model.vecmap_context_tgt_dico.word2id
        
    model.cuda()
    model.eval()
    s2t_dict = dict()

    # read muse dict
    with open(dict_path) as f:
        for line in f:
            sw, tw = line.rstrip().split(' ', 1)
            if sw not in s2t_dict:
                s2t_dict[sw] = [tw]
            else:
                s2t_dict[sw].append(tw)

    pre_1 = 0
    pre_5 = 0
    pre_10 = 0
    src_words = list(s2t_dict.keys())
    s_l = []
    s_w = []
    c_l = []
    t_l = []
    oov = 0
    for w in src_words:
        try:
            t_l = [s_tgt_w2i[wt] for wt in s2t_dict[w]]
            c_l.append(c_src_w2i[w])
            s_l.append(s_src_w2i[w])
            s_w.append(w)
        except:
            oov += 1
            pass

    static_src_id = torch.LongTensor(s_l)
    context_src_id = torch.LongTensor(c_l)

    if params.vecmap_context_src_emb_path != '' and params.vecmap_context_tgt_emb_path != '':
        vecmap_context_src_id = torch.LongTensor([vc_src_w2i[w] for w in s_w])

    tgt_ids = torch.zeros((len(s_w), 10)).cuda()
    for s in range(0, len(s_w), 1500):
        t = min(len(s_w), s+1500)
        if params.vecmap_context_src_emb_path != '' and params.vecmap_context_tgt_emb_path != '':
            tgt_ids[s:t] = model.test_all_word(static_src_id[s:t].cuda(), context_src_id[s:t].cuda(), vecmap_context_src_id[s:t].cuda(), params.vecmap, params.orign)
        else:
            tgt_ids[s:t] = model.test_all_word(static_src_id[s:t].cuda(), context_src_id[s:t].cuda(), None, params.vecmap, params.orign)
    

    for i in range(len(tgt_ids)):
        for j in range(10):
            if model.static_tgt_dico[tgt_ids[i][j].item()] in s2t_dict[s_w[i]]:
                if j == 0:
                    pre_1 += 1 
                    pre_5 += 1
                    pre_10 += 1
                    break
                elif j < 5:
                    pre_5 += 1
                    pre_10 += 1
                    break
                elif j < 10:
                    pre_10 += 1
                    break
    
    print('precision1 %f'%(pre_1/len(s_w)))
    print('precision5 %f'%(pre_5/len(s_w)))
    print('precision10 %f'%(pre_10/len(s_w)))
    print('courage:%f'%(len(s_w)/(len(s_w)+oov)))
print(params.src_lang, params.tgt_lang)
test(params.model_path, params.dict_path)
        
