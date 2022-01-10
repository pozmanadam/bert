#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from glob import glob
import torch
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pickle
import numpy as np
#pip install ipywidgets
#jupyter nbextension enable --py widgetsnbextension

# In[2]:


data_path = "/workDir/epochs"
model_names = ["hubert"]#, "bert-base-multilingual-cased"]


# In[3]:


files = {}
for model_name in model_names:
    f = sorted(glob(data_path + f"{model_name}-unfreezed-epoch*.*"), key=os.path.getmtime)
    files[model_name] = f
files


# In[4]:


def load_scores(model_path, map_location='cpu'):
    checkpoint = torch.load(model_path, map_location=map_location)
    return checkpoint['metrics']

def get_strict_f_score(report):
    #return sum(float(report['cls_report'][output]['f1-score']) for output in ('period', 'question', 'comma')) / 3
    return sum(float(report['cls_report'][output]['f1-score']) for output in ('period', 'question', 'comma', 'exclamation')) / 4

metrics = {}
for model_name in model_names:
    m = []
    for file in tqdm(files[model_name]):
        m.append(load_scores(file))
    metrics[model_name] = m
    
with open('reports/metrics_hun.pkl', 'wb') as f:
    pickle.dump(metrics, f)
    
with open('reports/metrics_hun.pkl', 'rb') as f:
    metrics = pickle.load(f)
    
for _, m in metrics.items():
    for epoch in m:
        epoch['strict_f_score'] = get_strict_f_score(epoch)
        
def best_epoch_by_f_score(metrics):
    best_score = metrics[0]['strict_f_score']
    best_epoch = 0
    for i, m in enumerate(metrics):
        if m['strict_f_score'] > best_score:
            best_score = m['strict_f_score']
            best_epoch = i
    return best_epoch, best_score

def best_epoch_by_loss(metrics):
    best_loss = metrics[0]['loss']
    best_epoch = 0
    for i, m in enumerate(metrics):
        if m['loss'] < best_loss:
            best_loss = m['loss']
            best_epoch = i
    return best_epoch, best_loss


# In[5]:


plt.style.use('seaborn-whitegrid')
# plt.title('Valid loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
for model_name, m in metrics.items():
    loss = [float(epoch['loss']) for epoch in m ]
    plt.plot(np.arange(len(loss))+1, loss, '--d')
plt.legend(model_names)
plt.savefig('/workDir/imgs/valid_loss_hun.pdf')
plt.show()

plt.style.use('seaborn-whitegrid')
# plt.title('Valid F1-score')
plt.ylabel('Macro $F_1$ score')
plt.xlabel('Epoch')
for model_name, m in metrics.items():
    f_score = [float(epoch['strict_f_score']) for epoch in m ]
    plt.plot(np.arange(len(loss))+1, f_score, '--d')
plt.legend(model_names)
plt.savefig('/workDir/imgs/valid_f1_score_hun.pdf')
plt.show()


# In[6]:


from neural_punctuator.utils.data import get_config_from_yaml
from neural_punctuator.models.BertPunctuator import BertPunctuator

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from neural_punctuator.data.dataloader import BertDataset, collate, get_data_loaders, get_datasets
from neural_punctuator.models.BertPunctuator import BertPunctuator
from torch.optim import AdamW
from torch import nn

from neural_punctuator.utils.io import save, load
from neural_punctuator.utils.metrics import get_total_grad_norm, get_eval_metrics
import numpy as np
import pickle

from torch.utils.data import Dataset, DataLoader
from itertools import product


# In[7]:


def combine(pred_num, preds):
    step_num = 512 // pred_num
    multi_preds = [preds[i::pred_num].reshape(-1, preds.shape[-1]) for i in range(pred_num)]
    for i in range(pred_num):
        start_idx = (pred_num - i - 1) * step_num
        end_idx = start_idx + (preds.shape[0] - (pred_num-1)*2) * step_num
        multi_preds[i] = multi_preds[i][start_idx:end_idx]

    multi_preds = np.stack(multi_preds)
    multi_preds = np.log(np.exp(multi_preds).mean(0))
    return multi_preds

def evaluate_multiple_predictions(model_name, model_type, predict_step, device, dataset_type):
    print(model_name, model_type)

    if model_type == 'by_f_score':
        epoch, _ = best_epoch_by_f_score(metrics[model_name])
    elif model_type == 'by_loss':
        epoch, _ = best_epoch_by_loss(metrics[model_name])
    else:
        raise ValueError("Model type not valid, options: by_f_score/by_loss")

    config = get_config_from_yaml(f'neural_punctuator/configs/config-{model_name}-unfreeze-ex.yaml')
    config.trainer.load_model = f"{model_name}-unfreezed-epoch-{epoch + 1}.pth"

    config.model.predict_step = predict_step
    config.predict.batch_size = 128
    

    model = BertPunctuator(config)
    model.to(device)

    load(model, None, config)

    test_dataset = BertDataset(dataset_type, config)

    test_loader = DataLoader(test_dataset, batch_size=config.predict.batch_size, collate_fn=collate)

    model.eval()
    all_test_preds = []

    for data in tqdm(test_loader):
        text, targets = data
        with torch.no_grad():
            preds, _ = model(text.to(device))

        all_test_preds.append(preds.detach().cpu().numpy())

    all_test_target = test_dataset.targets[512:-512]
    all_test_preds = np.concatenate(all_test_preds)
    pred_num = config.model.seq_len // config.model.predict_step

    ps = combine(pred_num, all_test_preds)
    _targets = np.array(all_test_target[:ps.shape[0]])

    ps = ps[_targets != -1]
    _targets = _targets[_targets != -1]

    report = get_eval_metrics(_targets, ps, config)
    return report


# In[8]:


class BertDataset(Dataset):
    def __init__(self, prefix, config, is_train=False):

        self.config = config
        self.is_train = is_train

        with open(self.config.data.data_path + prefix + "_data.pkl", 'rb') as f:
            texts, targets = pickle.load(f)
            self.encoded_texts = 512 * [0] + [word for t in texts for word in t] + 512 * [0]  # Add padding to both ends
            self.targets = 512 * [-1] + [t for ts in targets for t in ts] + 512 * [-1]

    def __getitem__(self, idx):
        if idx == 164:
            pass
        start_idx = (1+idx) * self.config.model.predict_step
        end_idx = start_idx + self.config.model.seq_len
        return torch.LongTensor(self.encoded_texts[start_idx: end_idx]),               torch.LongTensor(self.targets[start_idx: end_idx])

    def __len__(self):
        return int(np.ceil((len(self.encoded_texts)-1024)//self.config.model.predict_step))


# In[22]:


device = torch.device('cuda:3')
torch.cuda.set_device(device)

reports = {}


# In[24]:


for model_name, model_type in product(model_names, ('by_loss', 'by_f_score')):
    pred_num_for_token = 1
    while pred_num_for_token <= 64:
        predict_step = 512 // pred_num_for_token
        report = evaluate_multiple_predictions(model_name, model_type, predict_step, device, "valid")
        print(model_name, model_type, pred_num_for_token, get_strict_f_score(report))
        reports[(model_name, model_type, pred_num_for_token)] = report
        pred_num_for_token *=2


# In[25]:


reports


# In[35]:


with open('reports/valid_hu.pkl', 'wb') as f:
    pickle.dump(reports, f)

# with open('reports/valid_hu_hubert.pkl', 'wb') as f:
#     pickle.dump(reports, f)  
# with open('reports/valid_hu_bert_cased.pkl', 'wb') as f:
#     pickle.dump(reports, f)
# with open('reports/valid_hu_bert_uncased.pkl', 'wb') as f:
#     pickle.dump(reports, f)

# with open('reports/valid_hu_hubert.pkl', 'rb') as f:
#     reports_uncased = pickle.load(f)
# with open('reports/valid_hu_bert_cased.pkl', 'rb') as f:
#     reports_cased = pickle.load(f)
# with open('reports/valid_hu_bert_uncased.pkl', 'rb') as f:
#     reports_hubert = pickle.load(f)

# reports = {**reports_uncased, **reports_cased, **reports_hubert}


# In[36]:


best_pred_num_for_tokens = {}

for model_name, model_type in product(model_names, ('by_loss', 'by_f_score')):
    best_score = 0
    best_pred_num_for_token = 0
    
    pred_num_for_token = 1
    while pred_num_for_token <= 64:
        report = reports[(model_name, model_type, pred_num_for_token)]
        score = get_strict_f_score(report)
        
        if score > best_score:
            best_score = score
            best_pred_num_for_token = pred_num_for_token
        pred_num_for_token *=2
        
    best_pred_num_for_tokens[(model_name, model_type)] = (best_score, best_pred_num_for_token)
best_pred_num_for_tokens


# In[37]:


test_reports = []

for (model_name, model_type), (_, pred_num_for_token) in best_pred_num_for_tokens.items():
    if model_type == 'by_f_score':
        epoch, _ = best_epoch_by_f_score(metrics[model_name])
    elif model_type == 'by_loss':
        epoch, _ = best_epoch_by_loss(metrics[model_name])
    else:
        raise ValueError("Model type not valid, options: by_f_score/by_loss")
        
    predict_step = 512 // pred_num_for_token
    report = evaluate_multiple_predictions(model_name, model_type, predict_step, device, "test")
    print(model_name, model_type, pred_num_for_token, get_strict_f_score(report))
    test_reports.append((model_name, model_type, pred_num_for_token, epoch, get_strict_f_score(report), report))


# In[38]:


with open('reports/test_hu.pkl', 'wb') as f:
    pickle.dump(test_reports, f)


# In[42]:


print("Model name\t\tModel type\t# preds/token\tEpoch\tF non-empty\tF")

for model_name, model_type, pred_num_for_token, epoch, strict_f_score, report in test_reports:    
    print(f"{model_name:20}\t{model_type:10}\t"+
          f"{pred_num_for_token}\t\t{epoch}\t{strict_f_score*100:.1f}\t\t{report['f_score']*100:.1f}")
    
    print(" "*18 + "\t".join(('P', 'R', 'F')))
    for punc_type in ('comma', 'period', 'question','exclamation'):
        print(f"{punc_type:15}", end="")
        for metric_type in ('precision', 'recall', 'f1-score'):        
            print(f"\t{report['cls_report'][punc_type][metric_type]*100:.1f}", end="")
        print()
    print()


# # Plots for number of preds per token  selection

# In[40]:


scores = {}

for model_name, model_type in product(model_names, ('by_loss', 'by_f_score')):
    pred_num_for_token = 1
    s_ = []
    while pred_num_for_token <= 64:
        report = reports[(model_name, model_type, pred_num_for_token)]
        score = get_strict_f_score(report)
        s_.append(score)
        
        pred_num_for_token *=2
        
    scores[(model_name, model_type)] = s_


# In[41]:


for (model_name, model_type), f_scores in scores.items():
    print(model_name, model_type)
    plt.style.use('seaborn-whitegrid')
    # plt.title('Multiple predictions')
    plt.ylabel('Macro $F_1$ score')
    plt.xlabel('Number of predictions per token')
    plt.xticks(np.arange(int(np.log2(64))+1), [str(2**i) for i in range(0, int(np.log2(64))+1)])
    plt.plot(f_scores[::-1], '--d')
    plt.savefig(f'/workDir/imgs/valid_multiple_predictions/{model_name}_{model_type}.pdf')
    plt.show()


# In[ ]:




