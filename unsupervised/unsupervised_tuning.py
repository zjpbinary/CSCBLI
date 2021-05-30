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


parser = argparse.ArgumentParser(description='testing')
parser.add_argument("--model_path", type=str, default='model.pkl', help="point model path")
parser.add_argument("--src_lang", type=str, default='en', help="point src lang")
parser.add_argument("--tgt_lang", type=str, default='ar', help="point tgt lang")
parser.add_argument("--reload_src_ctx", type=str, default='', help="point tgt lang")
parser.add_argument("--reload_tgt_ctx", type=str, default='', help="point tgt lang")

params = parser.parse_args()


def test(model_path):

    weight = np.linspace(0, 1, 100).tolist()
    model = torch.load(model_path)


    s2t_dict = dict()

    if params.reload_src_ctx !="" and params.reload_tgt_ctx !="":
        model.reload_context(params.reload_src_ctx, params.reload_tgt_ctx, params.src_lang, params.tgt_lang)
    
    model.cuda()
    model.eval()
    
    for w in weight:
        aucc = model.unsupervise_fintune(w)
        print(f"{w:.2f}: {aucc*100:.2f}%")
test(params.model_path)
        