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
import os

torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1) 


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Discriminator training')
parser.add_argument("--src_lang", type=str, default='fr', help="point src lang")
parser.add_argument("--tgt_lang", type=str, default='en', help="point src lang")
parser.add_argument("--epoch", type=int, default=8, help="epoch")


parser.add_argument("--mode", type=str, default='add_orign_nw', help="point mode")

parser.add_argument("--static_src_emb_path", type=str, default='', help="point static_src_emb_path")
parser.add_argument("--static_tgt_emb_path", type=str, default='', help="point static_tgt_emb_path")

parser.add_argument("--context_src_emb_path", type=str, default='', help="point c_src_emb_path")
parser.add_argument("--context_tgt_emb_path", type=str, default='', help="point c_tgt_emb_path")

parser.add_argument("--vecmap_context_src_emb_path", type=str, default='', help="point vecmap_c_src_emb_path")
parser.add_argument("--vecmap_context_tgt_emb_path", type=str, default='', help="point vecmap_c_tgt_emb_path")

parser.add_argument("--train_data_path", type=str, default='', help="train data")

parser.add_argument("--save_path", type=str, default='model.pkl', help="point save path")

parser.add_argument("--negative", type=str2bool, default="True", help="")


params = parser.parse_args()
# print(params)
# os.makedirs(params.save_path, exist_ok=True)

model = Discriminator(params)
print(model)

# train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True, num_workers=2)
# optimizer = torch.optim.Adam(dis.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-08, weight_decay=0)

print('start train')

model.train()
model.cuda()


model.unsup_train(params.mode)

print('start save')

model.cpu()
torch.save(model, params.save_path+'_last')
