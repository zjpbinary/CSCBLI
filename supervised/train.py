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

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)  

parser = argparse.ArgumentParser(description='Discriminator training')
parser.add_argument("--src_lang", type=str, default='en', help="point src lang")
parser.add_argument("--tgt_lang", type=str, default='ar', help="point src lang")
parser.add_argument("--epoch", type=int, default=5, help="epoch")

parser.add_argument("--mode", type=str, default='add_orign_nw', help="point mode")

parser.add_argument("--static_src_emb_path", type=str, default='', help="point static_src_emb_path")
parser.add_argument("--static_tgt_emb_path", type=str, default='', help="point static_tgt_emb_path")

parser.add_argument("--context_src_emb_path", type=str, default='', help="point c_src_emb_path")
parser.add_argument("--context_tgt_emb_path", type=str, default='', help="point c_tgt_emb_path")

parser.add_argument("--vecmap_context_src_emb_path", type=str, default='', help="point vecmap_c_src_emb_path")
parser.add_argument("--vecmap_context_tgt_emb_path", type=str, default='', help="point vecmap_c_tgt_emb_path")

parser.add_argument("--train_data_path", type=str, default='', help="train data")

parser.add_argument("--save_path", type=str, default='model.pkl', help="point save path")

params = parser.parse_args()

dis = Discriminator(params)
print(dis)


train_data = MyDataset(
                    dis.static_src_dico.word2id,
                    dis.static_tgt_dico.word2id, 
                    dis.context_src_dico.word2id, 
                    dis.context_tgt_dico.word2id,
                    params.train_data_path,
                    )
train_loader = DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=2)


optimizer = torch.optim.Adam(dis.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-08, weight_decay=0)

print('start train')

dis.train()
for e in range(params.epoch):
    dis.cuda()
    print('start epoch %d'%e)
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        loss = dis.get_loss(data[0].cuda(), data[1].cuda(), data[2].cuda(), data[3].cuda(), data[4].cuda(), data[5].cuda(), params.mode)
        if i%100==0:
            print('start batch %d loss %f'%(i, loss))
        loss.backward()
        optimizer.step()
print('start save')
torch.save(dis.to('cpu'), params.save_path)
