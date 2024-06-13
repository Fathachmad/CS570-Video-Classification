import torch, os, sys, pickle
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from random import shuffle

sys.path.append('./custom_yolo/')
from customyolo import CustomYOLO

# 0 : plane, 1 : bird, 2 : clear
class YoloLstm(nn.Module):
    def __init__(self, module, augmented):
        super(YoloLstm, self).__init__()

        self.module = module
        self.augmented = augmented

        # Load train data
        self.data_zip, self.label, weight = self.data(True)

        # (N, L, 6) > (N, 100)
        self.lstm = nn.LSTM(input_size=6, hidden_size=50, batch_first=True, bidirectional=True, dtype=torch.float64)
        self.linear = nn.Linear(100, 3, bias=True, dtype=torch.float64)
        self.softmax = nn.Softmax(dim=-1)
        self.loss_fn = nn.CrossEntropyLoss(weight=weight)

        # Optimizer (Original lr = 0.0001)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.00005)

    def forward(self, data_zip, label):
        # Total prediction
        total_pred = torch.zeros(len(data_zip), 3, dtype=torch.float64)

        for data_ind, (len_tensor, track_tensor) in enumerate(data_zip):
            # No detection, so clear sky
            if len_tensor is None or track_tensor is None:
                total_pred[data_ind] = torch.tensor([0, 0, 1], dtype=torch.float64)
                continue

            # Run LSTM
            _, (lstm_res, _) = self.lstm(track_tensor)

            # Get weighted mean of LSTM outputs
            lstm_mean = torch.zeros(100)
            total_len = torch.sum(len_tensor)

            for i in range(len_tensor.size(0)):
                lstm_mean = lstm_mean + len_tensor[i] * torch.reshape(lstm_res[:, i, :], (-1, )) / total_len

            # Classification
            total_pred[data_ind] = self.softmax(self.linear(lstm_mean))

        # Evaluate result
        mat, correct, wrong = self.eval_pred(total_pred, label)
        loss = self.loss_fn(total_pred, label)
        
        return loss, (mat, correct, wrong)

    # Train function
    def train(self, num_epoch=200):
        eval_data_zip, eval_label, _ = self.data(False)

        for cur_epoch in range(num_epoch):
            loss, (mat, correct, wrong) = self.forward(self.data_zip, self.label)
            
            # Back propagation
            loss.backward()
            self.optimizer.step()

            if cur_epoch % 20 == 19:
                # Evaluation
                eval_loss, (eval_mat, eval_correct, eval_wrong) = self.forward(eval_data_zip, eval_label)

                print('Epoch : {} '.format(cur_epoch + 1))
                print('Train) loss : {}, matrix : {}, correct : {}, wrong : {}'.format(loss.item(), mat, correct, wrong))
                print('Eval) loss : {}, matrix : {}, correct : {}, wrong : {}'.format(eval_loss.item(), eval_mat, eval_correct, eval_wrong))

                # Save checkpoint
                torch.save({
                    'model_state_dict': self.state_dict(),
                    'loss': loss,
                    'train_result' : (loss.item(), mat, correct, wrong),
                    'eval_result' : (eval_loss.item(), eval_mat, eval_correct, eval_wrong)
                    }, './models/lower_lr_{}{}_epoch{}.pt'.format(self.module, '_augmented' if self.augmented else '', cur_epoch + 1))
    
    def data(self, train):
        length = []
        track = []
        label = []
        label_list = [] # For computing class weight

        for (root, dirs, files) in os.walk('../../data/data/{}/{}/{}/'.format(self.module,
        'data_augmented' if self.augmented else 'data',
        'train' if train else 'eval')):
            for file in files:
                file_path = os.path.join(root, file)
                label_path = root[root.rfind('/'):]

                # Append label data
                if 'airplane' in label_path:
                    label.append(torch.tensor([1, 0, 0], dtype=torch.float64))
                    label_list.append(0)
                elif 'bird' in label_path:
                    label.append(torch.tensor([0, 1, 0], dtype=torch.float64))
                    label_list.append(1)
                else:
                    label.append(torch.tensor([0, 0, 1], dtype=torch.float64))
                    label_list.append(2)

                # No object detected, as clear sky
                if os.path.getsize(file_path) == 0:
                    length.append(None)
                    track.append(None)
                    label_list.pop() # It's not trainable, so don't consider for training
                    continue

                # Track data exsists
                with open(file_path, 'rb') as pck:
                    length.append(pickle.load(pck))
                    track.append(pickle.load(pck))

        # Compute train_weight
        train_weight = compute_class_weight(class_weight="balanced", classes=np.unique(label_list), y=label_list)
        
        # No object detected for every clear sky data
        if len(train_weight) == 2:
            train_weight = np.append(train_weight, 0)

        return list(zip(length, track)), torch.stack(label), torch.tensor(train_weight, dtype=torch.float64)
    
    # Evaluate prediction
    def eval_pred(self, pred, true):
        mat = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        correct = 0
        wrong = 0

        pred_ind = torch.argmax(pred, dim=1)
        true_ind = torch.argmax(true, dim=1)

        for i in range(pred_ind.size(0)):
            pred_i = pred_ind[i]
            true_i = true_ind[i]

            mat[pred_i][true_i] = mat[pred_i][true_i] + 1
            if pred_i == true_i:
                correct = correct + 1
            else:
                wrong = wrong + 1

        return mat, correct, wrong