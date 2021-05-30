
import torch
import pdb
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, s_src_w2i, s_tgt_w2i, c_src_w2i, c_tgt_w2i, train_data_path):
    #def __init__(self, s_src_w2i, s_tgt_w2i, c_src_w2i, c_tgt_w2i):
        super(MyDataset, self).__init__()

        self.s_src_w2i = s_src_w2i
        self.s_tgt_w2i = s_tgt_w2i
        self.c_src_w2i = c_src_w2i
        self.c_tgt_w2i = c_tgt_w2i

        self.all_src_word, self.all_tgt_word, self.all_static_src_word_index, self.all_static_tgt_word_index, self.all_context_src_word_index, self.all_context_tgt_word_index = self.read_train_file(train_data_path)
    
    def __getitem__(self, index):
        return  self.data_index[index]
    
    def __len__(self):
        return len(self.all_tgt_word)
    def read_train_file(self, train_data_path):
        all_src_word = []
        all_tgt_word = []
        with open(train_data_path) as f:
            for line in f:
                ws, wt, _ = line.rstrip().split()
                if ws not in all_src_word:
                    all_src_word.append(ws)
                if wt not in all_tgt_word:
                    all_tgt_word.append(wt)
        all_static_src_word_index = [self.s_src_w2i[w] for w in all_src_word]
        all_static_tgt_word_index = [self.s_tgt_w2i[w] for w in all_tgt_word]
        all_context_src_word_index = [self.c_src_w2i[w] for w in all_src_word]
        all_context_tgt_word_index = [self.c_tgt_w2i[w] for w in all_tgt_word]

        return all_src_word, all_tgt_word, all_static_src_word_index, all_static_tgt_word_index, all_context_src_word_index, all_context_tgt_word_index

    