import os
import pandas as pd
import numpy as np
from pprint import pprint
from pathlib import Path
from collections import Counter
import pickle
import random
import argparse
import time
from tqdm import tqdm
from datetime import datetime
# torch:
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset,random_split
from torch.optim.lr_scheduler import ExponentialLR
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from transformers import BertTokenizer, AdamW, BertModel
from transformers import XLMTokenizer, XLMModel, XLMRobertaTokenizer, XLMRobertaModel
from tokenization_kobert import KoBertTokenizer

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import TensorBoardLogger

class Model(nn.Module):
    def __init__(self, embed_type):
        super().__init__()
        
        self.embed_type = embed_type
        
        if embed_type == 'kc':
            self.model = BertModel.from_pretrained("beomi/kcbert-base")
    
        elif embed_type == 'ko':
            self.model = BertModel.from_pretrained("monologg/kobert")
            
        elif embed_type == 'xlm':
            self.model = XLMRobertaModel.from_pretrained( "xlm-roberta-base")

        elif embed_type == 'mul':
            self.model = BertModel.from_pretrained("bert-base-multilingual-cased")
            

    def forward(self,data, **kwargs):
        outputs_data = self.model(input_ids =data, **kwargs)
        output = outputs_data[1]
        #output = outputs_data[0][:,1,:]  # 단어인경우 해당 토큰의 임베딩 저장!    
        return output
    
    def preprocess_dataframe(self):
        
        df = pd.read_pickle('emoji_edit.pkl')

        pprint(f"data Size: {len(df)}")
        self.dataset = TensorDataset(
        torch.tensor(df[f'{self.embed_type}_token'].tolist(), dtype=torch.long))

        dataloader = DataLoader(
                self.dataset,
                batch_size=2,
                shuffle=False
            )
        
        return dataloader,df

def main(embed_type,gpu): 
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    
    model = Model(embed_type).to(device)
    dataloader,df = model.preprocess_dataframe()
        
    for bi, inputs in tqdm(enumerate(dataloader)):
        embed =  model(inputs[0].to(device))
        embed_list.extend(embed.detach().cpu().numpy())
        
    df[f'{embed_type}_token_enc'] = embed_list   
    df.to_pickle('emoji_edit.pkl')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("main.py", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--embed_type", type=str, default='tuned')
    parser.add_argument("--gpu", type=str, default='0')
    
    args = parser.parse_args()
    print(args)

    embed_list=[]
    main(args.embed_type,args.gpu)
    
