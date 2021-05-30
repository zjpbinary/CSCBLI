
import torch
import pdb
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, s_src_w2i, s_tgt_w2i, c_src_w2i, c_tgt_w2i, train_data_path):
        super(MyDataset, self).__init__()

        self.s_src_w2i = s_src_w2i
        self.s_tgt_w2i = s_tgt_w2i
        self.c_src_w2i = c_src_w2i
        self.c_tgt_w2i = c_tgt_w2i
        # self.muse_out, self.pairs, self.all_data, self.data_index = self.read_file(train_data_path)
        self.data_index = self.read_train_file(train_data_path)
    def __getitem__(self, index):

        return  self.data_index[index]
    def __len__(self):
        return len(self.data_index)
    def read_train_file(self, train_data_path):
        with open(train_data_path) as f:
            all_e = [e.rstrip().split() for e in f.readlines()]
            
        all_data = []
        for i in range(len(all_e)//2):
            
            all_data.append((all_e[2*i][0], all_e[2*i][1], all_e[2*i+1][1]))

        all_data_index = []
        for e in all_data:
            try:
                all_data_index.append((self.s_src_w2i[e[0]], self.c_src_w2i[e[0]], self.s_tgt_w2i[e[1]], self.c_tgt_w2i[e[1]], self.s_tgt_w2i[e[2]], self.c_tgt_w2i[e[2]]))
            except:
                pass
        return all_data_index
    
