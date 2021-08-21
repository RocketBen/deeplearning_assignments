from utils import *
import models
import torch
from torch.utils import data
from tqdm import tqdm
ckpt_path = '/home/clb/PycharmProjects/deeplearning_assiments/Machine_Translation/transformer_3600_0.5366.pth'

device = torch.device('cuda:3')
val_set = MTDataset('val', force_pad=True)
val_loader = data.DataLoader(val_set, batch_size=1, collate_fn=val_set.collate_fn)
model = models.Transformer()
ckpt = torch.load(ckpt_path)
model.load_state_dict(ckpt['model'])
model = model.to(device)
print(model)
en_vocab = MTVocab('en')
zh_vocab = MTVocab('zh')

accuracy, val_loss, _ = model_test(model, device, val_loader)

print('val accuarcy: {} loss: {}'.format(accuracy, val_loss))
