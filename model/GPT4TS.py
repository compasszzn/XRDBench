from math import sqrt

import torch
import torch.nn as nn
import json
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer
from transformers import GPT2Config, GPT2Tokenizer, GPT2Model, AutoTokenizer
from layers.Embed import PatchEmbedding, ReplicationPad1d, DataEmbedding_wo_time

import transformers
from einops import rearrange

from layers.StandardNorm import Normalize
import torch.nn.functional as F
transformers.logging.set_verbosity_error()


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = 0
        self.seq_len = 8192
        self.patch_size = configs.patch_len
        self.stride = configs.stride
        self.gpt_layers = 6
        self.feat_dim = 1
        if configs.task == 'spg':
            self.num_classes = 230
        elif configs.task == 'crysystem':
            self.num_classes = 7
        self.d_model = 1280

        self.patch_num = (self.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        self.enc_embedding = DataEmbedding_wo_time(self.feat_dim * self.patch_size, self.d_model, dropout=configs.dropout)

        self.gpt2 = GPT2Model.from_pretrained('/data/LLM/gpt-2/774M', output_attentions=True, output_hidden_states=True, from_tf=True)
        self.gpt2.h = self.gpt2.h[:self.gpt_layers]
        
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        device = torch.device('cuda:{}'.format(0))
        self.gpt2.to(device=device)

        self.act = F.gelu
        self.dropout = nn.Dropout(0.1)
        self.ln_proj = nn.LayerNorm(self.d_model * self.patch_num)
        
        self.ln_proj = nn.LayerNorm(self.d_model * self.patch_num)
        self.out_layer = nn.Linear(self.d_model * self.patch_num, self.num_classes)


    def forward(self, x):
        x = F.interpolate(x,size=8192,mode='linear', align_corners=False)
        x = x.reshape(-1,8192,1)
        B, L, M = x.shape
        
        input_x = rearrange(x, 'b l m -> b m l')
        input_x = self.padding_patch_layer(input_x)
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        input_x = rearrange(input_x, 'b m n p -> b n (p m)')
        
        outputs = self.enc_embedding(input_x)
        
        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

        outputs = self.act(outputs).reshape(B, -1)
        outputs = self.ln_proj(outputs)
        outputs = self.out_layer(outputs)
        
        return outputs


