from utils import model_test
from utils import *
import models

ckpt_path = '/home/clb/PycharmProjects/deeplearning_assiments/Text_Classification/Rnn_8500_0.7501.pth'

device = torch.device('cuda:1')
epochs = 50
batch_size = 32

test_set = IMDBDataset(is_train=False, device=None)
test_loader = data.DataLoader(test_set, batch_size, collate_fn=test_set.collate_fn, shuffle=True)
model = models.RnnClassifier()
ckpt = torch.load(ckpt_path)
model.load_state_dict(ckpt['model'])
model = model.to(device)
print(model)
accu, test_loss = model_test(model, device=device, test_loader=test_loader)
print('inference:  accuracy: {}, ave_loss: {} '.format(accu, test_loss))