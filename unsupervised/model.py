import io
import numpy as np
import torch
import torch.nn as nn
import argparse
import pdb
import torch.nn.functional as F
from dictionary import Dictionary


np.random.seed(1)

class Discriminator(nn.Module):

    def __init__(self, params):
        super(Discriminator, self).__init__()

        self.params = params
        size = -1
        print("loading embedding...")
        self.static_src_dico, static_src_emb = self.read_txt_embeddings(params.static_src_emb_path, params.src_lang, size=size)
        self.static_tgt_dico, static_tgt_emb = self.read_txt_embeddings(params.static_tgt_emb_path, params.tgt_lang, size=size)
        self.static_src_embed = nn.Embedding(len(self.static_src_dico), 300)
        self.static_src_embed.weight.data.copy_(static_src_emb)
        self.static_src_embed.weight.requires_grad = False
        self.static_tgt_embed = nn.Embedding(len(self.static_tgt_dico), 300)
        self.static_tgt_embed.weight.data.copy_(static_tgt_emb)
        self.static_tgt_embed.weight.requires_grad = False
        print("loading embedding successfully")

        print("loading embedding...")
        self.context_src_dico, context_src_emb = self.read_txt_embeddings(params.context_src_emb_path, params.src_lang, size=size)
        self.context_tgt_dico, context_tgt_emb = self.read_txt_embeddings(params.context_tgt_emb_path, params.tgt_lang, size=size)
        self.context_src_embed = nn.Embedding(len(self.context_src_dico), 1024)
        self.context_src_embed.weight.data.copy_(context_src_emb)
        self.context_src_embed.weight.requires_grad = False
        self.context_tgt_embed = nn.Embedding(len(self.context_tgt_dico), 1024)
        self.context_tgt_embed.weight.data.copy_(context_tgt_emb)
        self.context_tgt_embed.weight.requires_grad = False
        print("loading embedding successfully")
        '''
        print("loading embedding...")
        self.vecmap_context_src_dico, vecmap_context_src_emb = self.read_txt_embeddings(params.vecmap_context_src_emb_path, params.src_lang)
        self.vecmap_context_tgt_dico, vecmap_context_tgt_emb = self.read_txt_embeddings(params.vecmap_context_tgt_emb_path, params.tgt_lang)
        self.vecmap_context_src_embed = nn.Embedding(len(self.vecmap_context_src_dico), 1024)
        self.vecmap_context_src_embed.weight.data.copy_(vecmap_context_src_emb)
        self.vecmap_context_src_embed.weight.requires_grad = False
        self.vecmap_context_tgt_embed = nn.Embedding(len(self.vecmap_context_tgt_dico), 1024)
        self.vecmap_context_tgt_embed.weight.data.copy_(vecmap_context_tgt_emb)
        self.vecmap_context_tgt_embed.weight.requires_grad = False
        print("loading embedding successfully")
        '''
        self.linear1 = nn.Linear(1024, 300)
        self.linear2 = nn.Linear(1024, 300)
        self.linear3 = nn.Linear(300, 300)
        self.linear4 = nn.Linear(300, 300)
        # dropout
        self.dropout = torch.nn.Dropout(p=0.1)
        

        self.w1 = nn.Parameter(torch.FloatTensor([0.]*300).cuda())
        self.w2 = nn.Parameter(torch.FloatTensor([0.]*300).cuda())
        self.save_path = params.save_path
        self.negative = params.negative


    def reload_context(self, path1, path2, src, tgt):
        print("loading embedding...")
        self.vecmap_context_src_dico, vecmap_context_src_emb = self.read_txt_embeddings(path1, src)
        self.vecmap_context_tgt_dico, vecmap_context_tgt_emb = self.read_txt_embeddings(path2, tgt)
        self.vecmap_context_src_embed = nn.Embedding(len(self.vecmap_context_src_dico), 1024)
        self.vecmap_context_src_embed.weight.data.copy_(vecmap_context_src_emb)
        self.vecmap_context_src_embed.weight.requires_grad = False
        self.vecmap_context_tgt_embed = nn.Embedding(len(self.vecmap_context_tgt_dico), 1024)
        self.vecmap_context_tgt_embed.weight.data.copy_(vecmap_context_tgt_emb)
        self.vecmap_context_tgt_embed.weight.requires_grad = False
        print("loading embedding successfully")
        return self.vecmap_context_src_dico, self.vecmap_context_src_embed, self.vecmap_context_tgt_dico, self.vecmap_context_tgt_embed


    @torch.no_grad()
    def test_all_word(self, static_src_id, context_src_id, vecmap_context_src_id=None):
        
        static_src_emb = self.static_src_embed(static_src_id)
        context_src_emb = self.context_src_embed(context_src_id)
        # vecmap_context_src_emb = self.vecmap_context_src_embed(vecmap_context_src_id)

        static_tgt_emb = self.static_tgt_embed.weight
        context_tgt_emb = self.context_tgt_embed.weight
        # vecmap_context_tgt_emb = self.vecmap_context_tgt_embed.weight


        static_src_emb = static_src_emb / static_src_emb.norm(2, 1, keepdim=True).expand_as(static_src_emb)
        # vecmap_context_src_emb = vecmap_context_src_emb / vecmap_context_src_emb.norm(2, 1, keepdim=True).expand_as(vecmap_context_src_emb)
        static_tgt_emb = static_tgt_emb / static_tgt_emb.norm(2, 1, keepdim=True).expand_as(static_tgt_emb)
        # vecmap_context_tgt_emb = vecmap_context_tgt_emb / vecmap_context_tgt_emb.norm(2, 1, keepdim=True).expand_as(vecmap_context_tgt_emb)
        

        context_src_emb = torch.tanh(self.linear1(context_src_emb))
        context_src_emb = torch.tanh(self.linear3(context_src_emb))
        

        context_tgt_emb = torch.tanh(self.linear2(context_tgt_emb))
        context_tgt_emb = torch.tanh(self.linear4(context_tgt_emb))

        sim = torch.matmul(static_src_emb + self.w1 * self.norm_center_norm(context_src_emb), static_tgt_emb.T + (self.w2* self.norm_center_norm(context_tgt_emb)).T)
        
        # sim = torch.matmul(static_src_emb, static_tgt_emb.T)
        
        # csls
        all_src_emb = self.static_src_embed.weight
        all_src_emb = all_src_emb / all_src_emb.norm(2, 1, keepdim=True).expand_as(all_src_emb)

        context_src_emb = self.context_src_embed.weight
        context_src_emb = torch.tanh(self.linear1(context_src_emb))
        context_src_emb = torch.tanh(self.linear3(context_src_emb))
        context_src_emb = self.norm_center_norm(context_src_emb)
        context_tgt_emb = self.norm_center_norm(context_tgt_emb)
        # bwd_sim = torch.matmul(static_tgt_emb, all_src_emb.T)self.norm
        bwd_sim = torch.zeros((1, static_tgt_emb.shape[0])).cuda()
        # shape: 1 x len(static_tgt_emb)
        bs = 100
        for i in range(0, static_tgt_emb.shape[0], bs):

            j = min(i + bs, static_tgt_emb.shape[0])
            bwd_sim[0, i:j] = self.topk_mean(torch.matmul(static_tgt_emb[i:j] + self.w2 * context_tgt_emb[i:j], all_src_emb.T + (self.w1 * context_src_emb).T), k=10)
            # bwd_sim[0, i:j] = self.topk_mean(torch.matmul(self.w2 * context_tgt_emb[i:j], (self.w1 * context_src_emb).T), k=10)
            # bwd_sim[0, i:j] = self.topk_mean(torch.matmul(static_tgt_emb[i:j], all_src_emb.T), k=10)
        sim = 2 * sim - bwd_sim
        tgt_ids = sim.topk(10, dim=1)

        return tgt_ids[1]
   
    @torch.no_grad()
    def test_all_wordV2(self, static_src_id, context_src_id, vecmap_context_src_id=None, lambda_w=0):
        
        static_src_emb = self.static_src_embed(static_src_id)
        context_src_emb = self.context_src_embed(context_src_id)
        vecmap_context_src_emb = self.vecmap_context_src_embed(vecmap_context_src_id)

        static_tgt_emb = self.static_tgt_embed.weight
        context_tgt_emb = self.context_tgt_embed.weight
        vecmap_context_tgt_emb = self.vecmap_context_tgt_embed.weight

        static_src_emb = static_src_emb / static_src_emb.norm(2, 1, keepdim=True).expand_as(static_src_emb)
        vecmap_context_src_emb = self.norm(vecmap_context_src_emb)
        static_tgt_emb = static_tgt_emb / static_tgt_emb.norm(2, 1, keepdim=True).expand_as(static_tgt_emb)
        vecmap_context_tgt_emb = self.norm(vecmap_context_tgt_emb)

        context_src_emb = torch.tanh(self.linear1(context_src_emb))
        context_src_emb = torch.tanh(self.linear3(context_src_emb))
        
        context_tgt_emb = torch.tanh(self.linear2(context_tgt_emb))
        context_tgt_emb = torch.tanh(self.linear4(context_tgt_emb))

        # sim = torch.matmul(static_src_emb + self.w1 * self.norm_center_norm(context_src_emb), static_tgt_emb.T + (self.w2* self.norm_center_norm(context_tgt_emb)).T)
        sim = torch.matmul(static_src_emb, static_tgt_emb.T)
        
        # csls
        all_src_emb = self.static_src_embed.weight
        all_src_emb = all_src_emb / all_src_emb.norm(2, 1, keepdim=True).expand_as(all_src_emb)

        context_src_emb = self.context_src_embed.weight
        context_src_emb = torch.tanh(self.linear1(context_src_emb))
        context_src_emb = torch.tanh(self.linear3(context_src_emb))
        context_src_emb = self.norm_center_norm(context_src_emb)
        context_tgt_emb = self.norm_center_norm(context_tgt_emb)
        # bwd_sim = torch.matmul(static_tgt_emb, all_src_emb.T)self.norm
        bwd_sim = torch.zeros((1, static_tgt_emb.shape[0])).cuda()
        # shape: 1 x len(static_tgt_emb)

        del self.static_tgt_embed, self.static_src_embed, self.context_tgt_embed, self.context_src_embed
        torch.cuda.empty_cache()

        bs = 100
        for i in range(0, static_tgt_emb.shape[0], bs):
            j = min(i + bs, static_tgt_emb.shape[0])
            # bwd_sim[0, i:j] = self.topk_mean(torch.matmul(static_tgt_emb[i:j] + self.w2 * context_tgt_emb[i:j], all_src_emb.T + (self.w1 * context_src_emb).T), k=10)
            bwd_sim[0, i:j] = self.topk_mean(torch.matmul(static_tgt_emb[i:j], all_src_emb.T), k=10)
        sim = 2 * sim - bwd_sim

        sim_2 = torch.matmul(vecmap_context_src_emb, vecmap_context_tgt_emb.T)
        vecmap_all_src_emb = self.vecmap_context_src_embed.weight
        vecmap_all_src_emb = self.norm(vecmap_all_src_emb)

        
        bwd_sim_2 = torch.zeros((1, static_tgt_emb.shape[0])).cuda()
        for i in range(0, static_tgt_emb.shape[0], bs):
            j = min(i + bs, static_tgt_emb.shape[0])
            bwd_sim_2[0, i:j] = self.topk_mean(torch.matmul(vecmap_context_tgt_emb[i:j], vecmap_all_src_emb.T), k=10)
    
        sim_2 = 2 * sim_2 - bwd_sim_2
        sim = sim + sim_2 * lambda_w

        tgt_ids = sim.topk(10, dim=1)

        return tgt_ids[1]


    @torch.no_grad()
    def unsupervise_fintune(self, w):
        size = 8000
        static_src_emb = self.static_src_embed.weight[:size]
        static_tgt_emb = self.static_tgt_embed.weight[:size]

        context_src_emb = self.context_src_embed.weight[:size]
        context_tgt_emb = self.context_tgt_embed.weight[:size]

        context_src_emb = torch.tanh(self.linear1(context_src_emb))
        context_src_emb = torch.tanh(self.linear3(context_src_emb))
        
        context_tgt_emb = torch.tanh(self.linear2(context_tgt_emb))
        context_tgt_emb = torch.tanh(self.linear4(context_tgt_emb))


        vecmap_context_src_emb = self.vecmap_context_src_embed.weight[:size]
        vecmap_context_tgt_emb = self.vecmap_context_tgt_embed.weight[:size]
        sim = torch.matmul(static_src_emb + self.w1 * self.norm_center_norm(context_src_emb) , static_tgt_emb.T + (self.w2 * self.norm_center_norm(context_tgt_emb)).T)
        sim -= self.topk_mean(sim, k=10).unsqueeze(-1) / 2 + self.topk_mean(sim.T, k=10).unsqueeze(0) / 2

        sim_y = sim.T

        sim1 = torch.matmul(vecmap_context_src_emb , vecmap_context_tgt_emb.T)
        sim1 -= self.topk_mean(sim1, k=10).unsqueeze(-1) / 2 + self.topk_mean(sim1.T, k=10).unsqueeze(0) / 2
        sim1_y = sim1.T

        sim = sim + sim1 * w
        sim_y = sim_y + sim1_y * w

        n_samples = 5000
        src_indices = torch.arange(0, n_samples).cuda()
        hyp_tgt_indices = sim[src_indices].topk(1)[1].squeeze(-1)
        hyp_src_indices = sim_y[hyp_tgt_indices].topk(1)[1].squeeze(1)
        aucc = (hyp_src_indices==src_indices).sum().float() / n_samples

        return aucc

    def unsup_train(self, mode):
        
        best_acc = 0
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-08, weight_decay=0)

        sy, ty, wy = mode.split('_')

        
        static_src_emb = self.static_src_embed.weight[:8000]
        static_tgt_emb = self.static_tgt_embed.weight[:8000]

        static_src_emb = static_src_emb / static_src_emb.norm(2, 1, keepdim=True).expand_as(static_src_emb)
        static_tgt_emb = static_tgt_emb / static_tgt_emb.norm(2, 1, keepdim=True).expand_as(static_tgt_emb)

        if ty=='orign':
            context_src_emb = self.context_src_embed.weight[:8000]
            context_tgt_emb = self.context_tgt_embed.weight[:8000]
        elif ty=='vecmap':
            context_src_emb = self.vecmap_context_src_embed.weight[:8000]
            context_tgt_emb = self.vecmap_context_tgt_embed.weight[:8000]
        # 加入dropout
        context_src_emb = self.dropout(context_src_emb)
        context_tgt_emb = self.dropout(context_tgt_emb)

    

        # context_src_emb = self.norm_center_norm(context_src_emb)    
        # context_tgt_emb = self.norm_center_norm(context_tgt_emb)    

        sim = torch.matmul(static_src_emb, static_tgt_emb.T)
        sim -= self.topk_mean(sim, k=10).unsqueeze(-1) / 2 + self.topk_mean(sim.T, k=10).unsqueeze(0) / 2

        if wy == 'w':
            vecmap_context_src_emb = self.vecmap_context_src_embed.weight[:8000]
            vecmap_context_tgt_emb = self.vecmap_context_tgt_embed.weight[:8000]
        '''
        self.matrix_s()
        vecmap_context_src_emb = vecmap_context_src_emb / vecmap_context_src_emb.norm(2, 1, keepdim=True).expand_as(vecmap_context_src_emb)
        vecmap_context_tgt_emb = vecmap_context_tgt_emb / vecmap_context_tgt_emb.norm(2, 1, keepdim=True).expand_as(vecmap_context_tgt_emb)
        context_sim = torch.matmul(vecmap_context_src_emb, vecmap_context_tgt_emb.T)
        context_sim -= self.topk_mean(context_sim, k=10).unsqueeze(-1) / 2 + self.topk_mean(context_sim.T, k=10).unsqueeze(0) / 2
        sim += 0.05 * context_sim 
        '''
        
        # union
        
        # src_wrong_indices = np.random.randint(1, 20, size=(len(sim), 1))
        # src_wrong_indices = torch.from_numpy(src_wrong_indices).cuda()
        # src_wrong_indices = torch.gather(torch.topk(sim, 20)[1], 1, src_wrong_indices)
        # src_indices = torch.cat([torch.arange(start=0, end=len(static_src_emb)).cuda().unsqueeze(-1), torch.argmax(sim, dim=1).unsqueeze(-1), src_wrong_indices], dim=1)
        # tgt_wrong_indices = np.random.randint(1, 20, size=(len(sim), 1))
        # tgt_wrong_indices = torch.from_numpy(tgt_wrong_indices).cuda()
        # tgt_wrong_indices = torch.gather(torch.topk(sim.T, 20)[1], 1, tgt_wrong_indices)
        # tgt_indices = torch.cat([torch.argmax(sim, dim=0).unsqueeze(-1), torch.arange(start=0, end=len(static_tgt_emb)).cuda().unsqueeze(-1), tgt_wrong_indices], dim=1)

        # start training loop
        print('start training loop')
        it = -1

        p = 0.1
        max_epoch = 1000
        cur_epoch = 0
        while cur_epoch < max_epoch:
            optimizer.zero_grad()
            # argmax Fs Fz
            
            topk = 100
            src_wrong_indices = np.random.randint(1, topk, size=(len(sim), 1))
            src_wrong_indices = torch.from_numpy(src_wrong_indices).cuda()
            src_wrong_indices = torch.gather(torch.topk(sim, topk)[1], 1, src_wrong_indices)
            src_indices = torch.cat([torch.arange(start=0, end=len(static_src_emb)).cuda().unsqueeze(-1), torch.argmax(sim, dim=1).unsqueeze(-1), src_wrong_indices], dim=1)
            
            tgt_wrong_indices = np.random.randint(1, topk, size=(len(sim), 1))
            tgt_wrong_indices = torch.from_numpy(tgt_wrong_indices).cuda()
            tgt_wrong_indices = torch.gather(torch.topk(sim.T, topk)[1], 1, tgt_wrong_indices)
            tgt_indices = torch.cat([torch.argmax(sim, dim=0).unsqueeze(-1), torch.arange(start=0, end=len(static_tgt_emb)).cuda().unsqueeze(-1), tgt_wrong_indices], dim=1)

            src_indices_src_context_emb = torch.tanh(self.linear1(context_src_emb[src_indices[:, 0]]))
            src_indices_tgt_context_emb = torch.tanh(self.linear2(context_tgt_emb[src_indices[:, 1]]))
            src_wrong_context_emb = torch.tanh(self.linear2(context_tgt_emb[src_indices[:, 2]]))
            tgt_indices_src_context_emb = torch.tanh(self.linear1(context_src_emb[tgt_indices[:, 0]]))
            tgt_indices_tgt_context_emb = torch.tanh(self.linear2(context_tgt_emb[tgt_indices[:, 1]]))
            tgt_wrong_context_emb = torch.tanh(self.linear1(context_src_emb[tgt_indices[:, 2]]))

            src_indices_src_context_emb = torch.tanh(self.linear3(src_indices_src_context_emb))
            src_indices_tgt_context_emb = torch.tanh(self.linear4(src_indices_tgt_context_emb))
            src_wrong_context_emb = torch.tanh(self.linear4(src_wrong_context_emb))
            tgt_indices_src_context_emb = torch.tanh(self.linear3(tgt_indices_src_context_emb))
            tgt_indices_tgt_context_emb = torch.tanh(self.linear4(tgt_indices_tgt_context_emb))
            tgt_wrong_context_emb = torch.tanh(self.linear3(tgt_wrong_context_emb))
           
            src_indices_src_context_emb = self.norm_center_norm(src_indices_src_context_emb)
            src_indices_tgt_context_emb = self.norm_center_norm(src_indices_tgt_context_emb)
            tgt_indices_src_context_emb = self.norm_center_norm(tgt_indices_src_context_emb)
            tgt_indices_tgt_context_emb = self.norm_center_norm(tgt_indices_tgt_context_emb)
            src_wrong_context_emb = self.norm_center_norm(src_wrong_context_emb)
            tgt_wrong_context_emb = self.norm_center_norm(tgt_wrong_context_emb)

            src_loss = torch.mul(static_src_emb[src_indices[:, 0]] + self.w1 * (src_indices_src_context_emb), static_tgt_emb[src_indices[:, 1]] + self.w2 * (src_indices_tgt_context_emb)).sum(dim=1) 
            tgt_loss = torch.mul(static_src_emb[tgt_indices[:, 0]] + self.w1 * (tgt_indices_src_context_emb), static_tgt_emb[tgt_indices[:, 1]] + self.w2 * (tgt_indices_tgt_context_emb)).sum(dim=1) 
            
            if self.negative:
                src_loss -= torch.mul(static_src_emb[src_indices[:, 0]] + self.w1 * (src_indices_src_context_emb), static_tgt_emb[src_indices[:, 2]] + self.w2 * (src_wrong_context_emb)).sum(dim=1)
                tgt_loss -= torch.mul(static_src_emb[tgt_indices[:, 2]] + self.w1 * (tgt_wrong_context_emb), static_tgt_emb[tgt_indices[:, 1]] + self.w2 * (tgt_indices_tgt_context_emb)).sum(dim=1)
                
            loss = 1 - src_loss.sum() / len(src_loss) - tgt_loss.sum() / len(tgt_loss)
            # loss = - src_loss.sum() / len(src_loss)

            loss.backward()
            optimizer.step()
            
            if p >= 1:
                break
            else:
                if it == -1:
                    pre_loss = loss.item()
                    it += 1 
                elif it >= 50:
                    p *= 2
                    it = 0
                    pre_loss = loss.item()
                elif it >= 0 and it < 50 and abs(loss.item() - pre_loss) <= 1e-6:
                    it += 1
                    pre_loss = loss.item()
                else:
                    pre_loss = loss.item()

            pre_src_indices = torch.zeros(src_indices.shape, requires_grad=False).cuda()
            pre_src_indices.data.copy_(src_indices)
            pre_tgt_indices = torch.zeros(tgt_indices.shape, requires_grad=False).cuda()
            pre_tgt_indices.data.copy_(tgt_indices)

            src_context_emb = torch.tanh(self.linear1(context_src_emb))
            tgt_context_emb = torch.tanh(self.linear2(context_tgt_emb))
            
            src_context_emb = torch.tanh(self.linear3(src_context_emb))
            tgt_context_emb = torch.tanh(self.linear4(tgt_context_emb))
            
            src_context_emb = self.norm_center_norm(src_context_emb)
            tgt_context_emb = self.norm_center_norm(tgt_context_emb)

            sim = torch.matmul(static_src_emb + self.w1 * src_context_emb , static_tgt_emb.T + (self.w2 * tgt_context_emb).T)
            sim -= self.topk_mean(sim, k=10).unsqueeze(-1) / 2 + self.topk_mean(sim.T, k=10).unsqueeze(0) / 2
            sim_y = sim.T

            n_samples = 4000
            src_indices = torch.arange(0, n_samples).cuda()
            hyp_tgt_indices = sim[src_indices].topk(1)[1].squeeze(-1)
            hyp_src_indices = sim_y[hyp_tgt_indices].topk(1)[1].squeeze(1)
            aucc = (hyp_src_indices==src_indices).sum().float() / n_samples
            print(f"loss: {pre_loss:.3f}   auccary: {aucc:.5f}  ")

            if best_acc <aucc:
                torch.save(self, self.save_path+"_best")
                best_acc = aucc

            if wy == 'w':
                vcs_emb = vecmap_context_src_emb
                vct_emb = vecmap_context_tgt_emb
                vcs_emb = self.norm(vcs_emb)
                vct_emb = self.norm(vct_emb)
                # vcs_emb = vcs_emb / vcs_emb.norm(2, 1, keepdim=True).expand_as(vcs_emb)
                # vct_emb = vct_emb / vct_emb.norm(2, 1, keepdim=True).expand_as(vct_emb)
                context_sim = torch.matmul(vcs_emb, vct_emb.T)
                context_sim -= self.topk_mean(context_sim, k=10).unsqueeze(-1) / 2 + self.topk_mean(context_sim.T, k=10).unsqueeze(0) / 2
                sim += 0.13 * context_sim

            src_indices = torch.cat([torch.arange(start=0, end=len(static_src_emb)).cuda().unsqueeze(-1), torch.argmax(sim, dim=1).unsqueeze(-1)], dim=1)
            tgt_indices = torch.cat([torch.argmax(sim, dim=0).unsqueeze(-1), torch.arange(start=0, end=len(static_tgt_emb)).cuda().unsqueeze(-1)], dim=1)

            src_indices_mask = pre_src_indices[:, 1] == src_indices[:, 1]
            # shape: len(src)
            tgt_indices_mask = pre_tgt_indices[:, 0] == tgt_indices[:, 0]

            src_change_mask = torch.rand(src_indices_mask.shape, requires_grad=False).cuda() >= 1 - p
            tgt_change_mask = torch.rand(tgt_indices_mask.shape, requires_grad=False).cuda() >= 1 - p
            
            src_change_indices =  (src_indices[:, 1] + torch.from_numpy(np.random.randint(len(static_tgt_emb), size=len(static_src_emb))).cuda() * (src_indices_mask & src_change_mask)) % len(static_tgt_emb)
            tgt_change_indices =  (tgt_indices[:, 0] + torch.from_numpy(np.random.randint(len(static_src_emb), size=len(static_tgt_emb))).cuda() * (tgt_indices_mask & tgt_change_mask)) % len(static_src_emb)

            src_wrong_indices = np.random.randint(1, 10, size=(len(sim), 1))
            src_wrong_indices = torch.from_numpy(src_wrong_indices).cuda()
            src_wrong_indices = torch.gather(torch.topk(sim, 10)[1], 1, src_wrong_indices)
            tgt_wrong_indices = np.random.randint(1, 10, size=(len(sim), 1))
            tgt_wrong_indices = torch.from_numpy(tgt_wrong_indices).cuda()
            tgt_wrong_indices = torch.gather(torch.topk(sim.T, 10)[1], 1, tgt_wrong_indices)

            src_indices = torch.cat([src_indices[:, 0].unsqueeze(-1), src_change_indices.unsqueeze(-1), src_wrong_indices], dim=1)
            tgt_indices = torch.cat([tgt_change_indices.unsqueeze(-1), tgt_indices[:, 1].unsqueeze(-1), tgt_wrong_indices], dim=1)
            
            cur_epoch += 1

        print('end train')
    
    def norm_center_norm(self, x):
        x = x / x.norm(2, 1, keepdim=True).expand_as(x)
        return x - torch.mean(x, dim=1).unsqueeze(-1)
    def norm(self, x):
        x = x / x.norm(2, 1, keepdim=True).expand_as(x)
        return x
    def topk_mean(self, m, k):

        ans, _ = torch.topk(m, k, dim=1)

        return ans.sum(1) / k

    def read_txt_embeddings(self, emb_path, lang, size=-1):
        """
        Reload pretrained embeddings from a text file.
        """
        word2id = {}
        vectors = []
        max_size = 100000000 if size == -1 else size
        # load pretrained embeddings
        with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
            for i, line in enumerate(f):
                if i == 0:
                    split = line.split()
                    assert len(split) == 2
                else:
                    word, vect = line.rstrip().split(' ', 1)
                    word = word.lower()
                    vect = np.fromstring(vect, sep=' ', dtype='float')
                    if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                        vect[0] = 0.01
                    if word in word2id:
                        
                        print('word have existed')
                    else:
                        word2id[word] = len(word2id)
                        vectors.append(vect[None])
                if i == size:
                    break

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



