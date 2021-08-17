import sys
sys.path.append("../")

# KoBERT 모델

import config

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
from tqdm import tqdm, tqdm_notebook



from KoBERT.kobert.utils import get_tokenizer
from KoBERT.kobert.pytorch_kobert import get_pytorch_kobert_model

from transformers import AdamW
# from transformers.optimization import WarmupLinearSchedule

from transformers import get_linear_schedule_with_warmup

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bertmodel, vocab = get_pytorch_kobert_model()

# 토크나이저 메서드를 tokenizer에 호출
# 코퍼스를 토큰으로 만드는 과정을 수행, 이 때 토크나이저는 kobert 패키지에 있는 get_tokenizer()를 사용하고,
# 토큰화를 위해 필요한 단어 사전은 kobert의 vocab을 사용함.
# uncased로 투입해야 하므로 lower = False

tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower = False)
print(f'device using: {device}')


model_config=config.model_config


class Data_for_BERT(Dataset):
    def __init__(self, dataset, max_len, pad, pair, label_cols):

        # gluon nlp 패키지의 data.BERTSentenceTransform 메서드를 사용,
        # 버트 활용을 위한 토크나이저를 bert_tokenizer로 주고,
        # 문장 내 시퀀스의 최대 수를 max_len 인자로 제공. 이 말은 max_len개의 (단어를 쪼갠) 덩어리만 활용한다는 의미
        # pad 인자는 max_len보다 짧은 문장을 패딩해주겠냐는 것을 묻는 것,
        # pair 인자는 문장으로 변환할지, 문장 쌍으로 변환할지.
        
        transform = nlp.data.BERTSentenceTransform(tok, max_seq_length = max_len, pad=pad, pair=pair)
        self.sentences = [transform([txt]) for txt in dataset.text]
        # self.sentences_Customer = [transform([txt]) for txt in dataset.Customer]
        # self.labels = [np.int32(i) for i in dataset.label]
        self.labels=dataset[label_cols].values

        # ohe = OneHotEncoder().fit(pd.Series(self.labels).values.reshape(-1,1))
        # self.labels = ohe.transform(pd.Series(self.labels).values.reshape(-1,1)).toarray()

        # target.bcat
        # self.labels = b_ohe.fit_transform(pd.Series(self.labels).values.reshape(-1,1))

    def __getitem__(self,i):
        return (self.sentences[i] + (self.labels[i],))
    
    def __len__(self):
        return(len(self.labels))


class BERTClassifier(nn.Module):
    def __init__(self, hidden_size = 768, num_classes = 8, dr_rate = None, params = None):
        # BERTClassifier의 자식 노드에 클래스 속성을 상속시켜준다?
        # 인풋으로 넣는 bert 모델을 클래스 내부의 bert 메서드로 넣어주고
        # dr_rate(drop-out rate)를 dr_rate로 넣어줌
        
        super(BERTClassifier, self).__init__()
        self.bert = bertmodel
        self.dr_rate = dr_rate
        
        # 여기서 nn.Linear는 keras의 Dense 층과 같은 fully connected layer
        # hidden layer의 사이즈를 입력해주고(여기서는 768)
        # out-put layer의 사이즈를 num_classes 인자의 수만큼 잡아줌.
        
#         self.lstm_layer = nn.LSTM(512, 128, 2)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        # dr_rate가 정의되어 있을 경우, 넣어준 비율에 맞게 weight를 drop-out 시켜줌
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
            
    def generate_attention_mask(self, token_ids, valid_length):
        
        # 버트 모델에 사용할 attention_mask를 만들어 줌.
        # token_id를 인풋으로 받아, attention mask를 만들어 냄
        
        # torch.zeros_like()는 토치 텐서를 인풋으로 받아, 스칼라 값 0으로 채워진 같은 사이즈의 텐서를 뱉어냄
        attention_mask = torch.zeros_like(token_ids)
        
        for i,v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()
    
    def forward(self, token_ids, valid_length, segment_ids):
        #  attention mask 를 만들어 내고, 버트 모델을 넣어줌. 
        attention_mask = self.generate_attention_mask(token_ids, valid_length)
        
        # .long() pytorch는  .to()와 같은 기능을 수행함. 이는 장치(GPU)에 모델을 넣어주는 역할을 수행
        # 출력값으로 classifier()
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out=self.dropout(pooler)
        
#         output=self.lstm_layer(out)
        
        return self.classifier(out)
    

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
