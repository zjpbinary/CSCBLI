import io
import numpy as np
import torch
import torch.nn as nn
import argparse
import pdb
import torch.nn.functional as F
from dictionary import Dictionary

class Discriminator(nn.Module):

    def __init__(self, params):
        super(Discriminator, self).__init__()

        self.params = params
        self.static_src_dico, static_src_emb = self.read_txt_embeddings(params.static_src_emb_path, params.src_lang)
        self.static_tgt_dico, static_tgt_emb = self.read_txt_embeddings(params.static_tgt_emb_path, params.tgt_lang)
        self.static_src_embed = nn.Embedding(len(self.static_src_dico), 300)
        self.static_src_embed.weight.data.copy_(static_src_emb)
        self.static_src_embed.weight.requires_grad = False
        self.static_tgt_embed = nn.Embedding(len(self.static_tgt_dico), 300)
        self.static_tgt_embed.weight.data.copy_(static_tgt_emb)
        self.static_tgt_embed.weight.requires_grad = False

        self.context_src_dico, context_src_emb = self.read_txt_embeddings(params.context_src_emb_path, params.src_lang)
        self.context_tgt_dico, context_tgt_emb = self.read_txt_embeddings(params.context_tgt_emb_path, params.tgt_lang)
        self.context_src_embed = nn.Embedding(len(self.context_src_dico), 1024)
        self.context_src_embed.weight.data.copy_(context_src_emb)
        self.context_src_embed.weight.requires_grad = False
        self.context_tgt_embed = nn.Embedding(len(self.context_tgt_dico), 1024)
        self.context_tgt_embed.weight.data.copy_(context_tgt_emb)
        self.context_tgt_embed.weight.requires_grad = False
        '''
        self.vecmap_context_src_dico, vecmap_context_src_emb = self.read_txt_embeddings(params.vecmap_context_src_emb_path, params.src_lang)
        self.vecmap_context_tgt_dico, vecmap_context_tgt_emb = self.read_txt_embeddings(params.vecmap_context_tgt_emb_path, params.tgt_lang)
        self.vecmap_context_src_embed = nn.Embedding(len(self.vecmap_context_src_dico), 1024)
        self.vecmap_context_src_embed.weight.data.copy_(vecmap_context_src_emb)
        self.vecmap_context_src_embed.weight.requires_grad = False
        self.vecmap_context_tgt_embed = nn.Embedding(len(self.vecmap_context_tgt_dico), 1024)
        self.vecmap_context_tgt_embed.weight.data.copy_(vecmap_context_tgt_emb)
        self.vecmap_context_tgt_embed.weight.requires_grad = False
        '''

        self.linear1 = nn.Linear(1024, 300)
        self.linear2 = nn.Linear(1024, 300)

        self.linear3 = nn.Linear(300, 300)
        self.linear4 = nn.Linear(300, 300)

        self.dropout = torch.nn.Dropout(p=0.2)

        self.ws = 0.1
        self.wt = 0.1

    @torch.no_grad()
    def test_all_word(self, static_src_id, context_src_id, vecmap_context_src_id=None, vecmap=False, orign=False):
        
        static_src_emb = self.static_src_embed(static_src_id)
        context_src_emb = self.context_src_embed(context_src_id)
        

        static_tgt_emb = self.static_tgt_embed.weight
        context_tgt_emb = self.context_tgt_embed.weight

        static_src_emb = static_src_emb / static_src_emb.norm(2, 1, keepdim=True).expand_as(static_src_emb)
        static_tgt_emb = static_tgt_emb / static_tgt_emb.norm(2, 1, keepdim=True).expand_as(static_tgt_emb)

        #vecmap_context_src_emb = self.norm_center_norm(vecmap_context_src_emb)
        #vecmap_context_tgt_emb = self.norm_center_norm(vecmap_context_tgt_emb)
         

        context_src_emb = torch.tanh(self.linear1(context_src_emb))
        context_src_emb = torch.tanh(self.linear3(context_src_emb))

        context_tgt_emb = torch.tanh(self.linear2(context_tgt_emb))
        context_tgt_emb = torch.tanh(self.linear4(context_tgt_emb))
        # pdb.set_trace()
        if not orign:
            sim = torch.matmul(static_src_emb + self.ws * self.norm(context_src_emb), static_tgt_emb.transpose(0, 1) + (self.wt * self.norm(context_tgt_emb)).transpose(0, 1))
        else:
            sim = torch.matmul(static_src_emb, static_tgt_emb.transpose(0, 1))
        
        all_src_emb = self.static_src_embed.weight
        all_src_emb = all_src_emb / all_src_emb.norm(2, 1, keepdim=True).expand_as(all_src_emb)

        context_src_emb = self.context_src_embed.weight
        context_src_emb = torch.tanh(self.linear1(context_src_emb))
        context_src_emb = torch.tanh(self.linear3(context_src_emb))
        context_src_emb = self.norm(context_src_emb)
        context_tgt_emb = self.norm(context_tgt_emb)
        # bwd_sim = torch.matmul(static_tgt_emb, all_src_emb.transpose(0, 1))

        bwd_sim = torch.zeros((1, static_tgt_emb.shape[0])).cuda()
        # shape: 1 x len(static_tgt_emb)
        bs = 100
        for i in range(0, static_tgt_emb.shape[0], bs):
            j = min(i + bs, static_tgt_emb.shape[0])
            # pdb.set_trace()
            if not orign:
                bwd_sim[0, i:j] = self.topk_mean(torch.matmul(static_tgt_emb[i:j] + self.wt * context_tgt_emb[i:j], all_src_emb.transpose(0, 1) + (self.ws * context_src_emb).transpose(0, 1)), k=10)
            else:
                bwd_sim[0, i:j] = self.topk_mean(torch.matmul(static_tgt_emb[i:j], all_src_emb.transpose(0, 1)), k=10)
        sim = 2 * sim - bwd_sim

        torch.cuda.empty_cache()

        if vecmap == True:
            self.vecmap_context_src_embed.weight.data.copy_(self.norm_center_norm(self.vecmap_context_src_embed.weight.data))
            vecmap_context_src_emb = self.vecmap_context_src_embed(vecmap_context_src_id)
            vecmap_context_tgt_emb = self.norm_center_norm(self.vecmap_context_tgt_embed.weight.data)

            sim_2 = torch.matmul(vecmap_context_src_emb, vecmap_context_tgt_emb.transpose(0, 1))
            all_src_emb = self.vecmap_context_src_embed.weight

            #all_src_emb = self.norm_center_norm(all_src_emb)
            
            bwd_sim = torch.zeros((1, static_tgt_emb.shape[0])).cuda()
            for i in range(0, static_tgt_emb.shape[0], bs):
                j = min(i + bs, static_tgt_emb.shape[0])
                bwd_sim[0, i:j] = self.topk_mean(torch.matmul(vecmap_context_tgt_emb[i:j], all_src_emb.transpose(0, 1)), k=10)
            sim_2 = 2 * sim_2 - bwd_sim

            sim = sim + sim_2 * 0.05
        
        tgt_ids = sim.topk(10, dim=1)
        
        return tgt_ids[1]

    def forward(self, static_src_id, context_src_id, static_tgt_id, context_tgt_id, mode):

        static_src_emb = self.static_src_embed(static_src_id)
        static_tgt_emb = self.static_tgt_embed(static_tgt_id)
        static_src_emb = static_src_emb / static_src_emb.norm(2, 1, keepdim=True).expand_as(static_src_emb)
        static_tgt_emb = static_tgt_emb / static_tgt_emb.norm(2, 1, keepdim=True).expand_as(static_tgt_emb)
        sy, ty, wy = mode.split('_')

        if ty=='orign':
            context_src_emb = self.context_src_embed(context_src_id)
            context_tgt_emb = self.context_tgt_embed(context_tgt_id)
        elif ty=='vecmap':
            context_src_emb = self.vecmap_context_src_embed(context_src_id)
            context_tgt_emb = self.vecmap_context_tgt_embed(context_tgt_id)

        '''
        context_src_emb = static_src_emb
        context_tgt_emb = static_tgt_emb
        '''

        context_src_emb = torch.tanh(self.linear1(context_src_emb))
        context_tgt_emb = torch.tanh(self.linear2(context_tgt_emb))

        context_src_emb = torch.tanh(self.linear3(context_src_emb))
        context_tgt_emb = torch.tanh(self.linear4(context_tgt_emb))

        context_src_emb = self.norm(context_src_emb)
        context_tgt_emb = self.norm(context_tgt_emb)
        # context_src_emb = context_src_emb / context_src_emb.norm(2, 1, keepdim=True).expand_as(context_src_emb)
        # context_tgt_emb = context_tgt_emb / context_tgt_emb.norm(2, 1, keepdim=True).expand_as(context_tgt_emb)
        
        if sy=='add':
            src_emb = static_src_emb + context_src_emb * self.ws
            tgt_emb = static_tgt_emb + context_tgt_emb * self.wt
        elif sy=='dot':

            context_src_emb = context_src_emb * 0.02 + 1
            context_tgt_emb = context_tgt_emb * 0.02 + 1
            src_emb = static_src_emb.mul(context_src_emb)
            tgt_emb = static_tgt_emb.mul(context_tgt_emb)
        
        src_all_emb = self.static_src_embed.weight / self.static_src_embed.weight.norm(2, 1, keepdim=True).expand_as(self.static_src_embed.weight)
        src_all_context_emb = torch.tanh(self.linear3(torch.tanh(self.linear1(self.context_src_embed.weight))))
        src_all_context_emb = self.norm(src_all_context_emb)

        # src_all_context_emb = src_all_context_emb / src_all_context_emb.norm(2, 1, keepdim=True).expand_as(src_all_context_emb)

        similarities_1 = 2 * src_emb.mul(tgt_emb).sum(1) - self.topk_mean(tgt_emb.mm(self.static_src_embed.weight.transpose(0, 1) + (self.ws * src_all_context_emb).transpose(0,1)), 10)

        # baseline
        # similarities_1 = 2 * static_src_emb.mul(static_tgt_emb).sum(1) - self.topk_mean(static_tgt_emb.mm(src_all_emb.transpose(0,1 )), 10)

        return similarities_1
        
    
    def topk_mean(self, m, k):
        # k int: neighbor
        ans, _ = torch.topk(m, k, dim=1)

        return ans.sum(1) / k

    def get_loss(self, static_src_id, context_src_id, static_right_tgt_id, context_right_tgt_id, static_wrong_tgt_id, context_wrong_tgt_id, mode):       
        s1 = self.forward(static_src_id, context_src_id, static_right_tgt_id, context_right_tgt_id, mode)
        # s2 = self.forward(static_src_id, context_src_id, static_wrong_tgt_id, context_wrong_tgt_id, mode)
        # loss = s2-s1
        loss = -s1
        return loss.sum()

    def norm(self, x):
        
        return x / x.norm(2, 1, keepdim=True).expand_as(x)

    def norm_center_norm(self, x):
        x = x.cpu().numpy()
        x /= np.linalg.norm(x, axis=1)[:, np.newaxis] + 1e-8
        x -= x.mean(axis=0)[np.newaxis, :]
        x /= np.linalg.norm(x, axis=1)[:, np.newaxis] + 1e-8
        '''
        x /= x.norm(2, 1, keepdim=True).expand_as(x) + 1e-8
        # pdb.set_trace()
        x = x - torch.mean(x, dim=0).unsqueeze(0)
        x /= x.norm(2, 1, keepdim=True).expand_as(x) + 1e-8
        '''
        return torch.from_numpy(x).cuda()

    def read_txt_embeddings(self, emb_path, lang):
        """
        Reload pretrained embeddings from a text file.
        """
        word2id = {}
        vectors = []

        # load pretrained embeddings
        with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
            for i, line in enumerate(f):
                if i == 0:
                    split = line.split()
                    assert len(split) == 2
                    
                else:
                    word, vect = line.rstrip().split(' ', 1)
                    word = word.lower()
                    vect = np.fromstring(vect, sep=' ')
                    if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                        vect[0] = 0.01
                    if word in word2id:

                        print('word have existed')
                    else:
                        word2id[word] = len(word2id)
                        vectors.append(vect[None])

        assert len(word2id) == len(vectors)
        
        # compute new vocabulary / embeddings
        id2word = {v: k for k, v in word2id.items()}
        dico = Dictionary(id2word, word2id, lang)
        embeddings = np.concatenate(vectors, 0)
        embeddings = torch.from_numpy(embeddings).float()

        return dico, embeddings  

    def read_txt_embeddings_bpe_v2(self, emb_path, lang):
        word2id = dict()

        bpe_checkpoint = torch.load(emb_path)
        word2tensor = bpe_checkpoint['bpe_vec']
        word2id = {k: i for i, k in enumerate(word2tensor.keys())}
        id2word = {v: k for k, v in word2id.items()}
        id2tensor = {word2id[k]:v for k, v in word2tensor.items()}

        dico = Dictionary(id2word, word2id, lang)
        return dico, id2tensor    

    def reload_vecmap_context_emb(self, context_src_emb_path, context_tgt_emb_path):

        self.vecmap_context_src_dico, vecmap_context_src_emb = self.read_txt_embeddings(context_src_emb_path, self.params.src_lang)
        self.vecmap_context_tgt_dico, vecmap_context_tgt_emb = self.read_txt_embeddings(context_tgt_emb_path, self.params.tgt_lang)
        self.vecmap_context_src_embed = nn.Embedding(len(self.vecmap_context_src_dico), 1024)
        self.vecmap_context_src_embed.weight.data.copy_(vecmap_context_src_emb)
        self.vecmap_context_src_embed.weight.requires_grad = False
        self.vecmap_context_tgt_embed = nn.Embedding(len(self.vecmap_context_tgt_dico), 1024)
        self.vecmap_context_tgt_embed.weight.data.copy_(vecmap_context_tgt_emb)
        self.vecmap_context_tgt_embed.weight.requires_grad = False

    def reload_static_emb(self, static_src_emb_path, static_tgt_emb_path):

        self.static_src_dico, static_src_emb = self.read_txt_embeddings(static_src_emb_path, self.params.src_lang)
        self.static_tgt_dico, static_tgt_emb = self.read_txt_embeddings(static_tgt_emb_path, self.params.tgt_lang)
        self.static_src_embed = nn.Embedding(len(self.static_src_dico), 300)
        self.static_src_embed.weight.data.copy_(static_src_emb)
        self.static_src_embed.weight.requires_grad = False
        self.static_tgt_embed = nn.Embedding(len(self.static_tgt_dico), 300)
        self.static_tgt_embed.weight.data.copy_(static_tgt_emb)
        self.static_tgt_embed.weight.requires_grad = False

    def reload_context_emb(self, context_src_emb_path, context_tgt_emb_path):

        self.context_src_dico, context_src_emb = self.read_txt_embeddings(context_src_emb_path, self.params.src_lang)
        self.context_tgt_dico, context_tgt_emb = self.read_txt_embeddings(context_tgt_emb_path, self.params.tgt_lang)
        self.context_src_embed = nn.Embedding(len(self.context_src_dico), 1024)
        self.context_src_embed.weight.data.copy_(context_src_emb)
        self.context_src_embed.weight.requires_grad = False
        self.context_tgt_embed = nn.Embedding(len(self.context_tgt_dico), 1024)
        self.context_tgt_embed.weight.data.copy_(context_tgt_emb)
        self.context_tgt_embed.weight.requires_grad = False

