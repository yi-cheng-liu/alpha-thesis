import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import wandb

class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ResBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class PolicyHead(nn.Module):
    def __init__(self, in_size, action_size):
        super(PolicyHead, self).__init__()
        self.fc1 = nn.Linear(in_size, 256)
        self.fc_bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(256, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc_bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

class ValueHead(nn.Module):
    def __init__(self, in_size, out_size):
        super(ValueHead, self).__init__()
        self.fc1 = nn.Linear(in_size, 256)
        self.fc_bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(256, out_size)
    
    def forward(self, x):
        x = F.relu(self.fc_bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.relu(x)

class NNetWrapper(nn.Module):
    def __init__(self, game):
        super(NNetWrapper, self).__init__()
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.num_channels = 128
        self.dropout = 0.3
        self.epochs = 5
        self.batch_size = 64

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.input_layer = nn.Conv2d(1, self.num_channels, kernel_size=3, padding=1).to(self.device)
        self.bn = nn.BatchNorm2d(self.num_channels).to(self.device)
        self.res_blocks = nn.Sequential(*[ResBlock(self.num_channels) for _ in range(8)]).to(self.device)

        self.policy_head = PolicyHead(self.board_y*self.board_x*self.num_channels, self.action_size).to(self.device)
        self.value_head = ValueHead(self.board_y*self.board_x*self.num_channels, 3).to(self.device)
        self.action_head = ValueHead(self.board_y*self.board_x*self.num_channels, 1).to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters())
        
    def forward(self, x):
        x = x.view(-1, 1, self.board_y, self.board_x)  # Reshape input to (batch_size, channels, H, W)
        x = F.relu(self.bn(self.input_layer(x)))
        x = self.res_blocks(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layers

        pi = self.policy_head(x)
        v = self.value_head(x)
        q = self.action_head(x)

        return pi, v, q

    def predict(self, board):
        """
        Predicts pi and v for one board state.
        """
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            board = torch.tensor(board, dtype=torch.float32).to(self.device)
            pi, v, q = self(board)
            self.train()  # Set the model back to training mode
            return pi.data.cpu().numpy(), v.data.cpu().numpy(), q.data.cpu().numpy()
    
    def predict_parallel(self, boards):
        return self.predict(boards)
    
    def loss_pi(self, targets, outputs):
        # return -torch.sum(targets * outputs) / targets.size()[0]
        return torch.sum(-targets*(outputs+1e-5).log(), dim=1).mean()
    
    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs) ** 2) / targets.size()[0]
    
    def loss_q(self, targets, outputs):
        return torch.sum((targets + outputs) ** 2) / targets.size()[0]
    
    def loss_Lp(self, v, q):
        min_q = torch.min(q, dim=1)[0]  # shape: (batch,)
        q_v_sum = min_q + v[:, 0]  # shape: (batch,)
        squared_diff = torch.square(q_v_sum)
        return torch.sum(squared_diff)
    
    def loss_Lq(self, v, q):
        # TODO: is z_s the same as v_0?
        v_0 = v[:, 0] # shape: (batch,)
        max_term = torch.max(-v_0, torch.zeros_like(v_0))  # shape: (batch,)
        v_q_sum = v_0.view(-1, 1) + q  # shape: (batch, action_size)
        squared_sum = torch.sum(torch.square(v_q_sum))  # shape: (batch,)
        return torch.sum(max_term / self.action_size * squared_sum)
    
    def fit(self, examples):
        """
        Trains the model using the examples.
        """
        for epoch in range(self.epochs):
            print("EPOCH ::: " + str(epoch + 1))
            self.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            q_losses = AverageMeter()
            # lp_losses = AverageMeter()
            # lq_losses = AverageMeter()

            batch_count = int(len(examples) / self.batch_size)
            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=self.batch_size)
                boards, pis, vs, qs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64)).to(self.device)
                target_pis = torch.FloatTensor(np.array(pis)).to(self.device)
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64)).to(self.device)
                target_qs = torch.FloatTensor(np.array(qs).astype(np.float64)).to(self.device)

                # compute output
                out_pi, out_v, out_q = self(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                l_q = self.loss_q(target_vs, out_q)
                # l_Lp = self.loss_Lp(out_v, out_q)
                # l_Lq = self.loss_Lq(target_vs, out_q)
                total_loss = l_pi + l_v + l_q # + l_Lp + l_Lq

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                q_losses.update(l_q.item(), boards.size(0))
                # lp_losses.update(l_Lp.item(), boards.size(0))
                # lq_losses.update(l_Lq.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses, Loss_q=q_losses) #, Loss_Lp=lp_losses, Loss_Lq=lq_losses)
                wandb.log({"Loss_pi": pi_losses.avg, "Loss_v": v_losses.avg, "Loss_q": q_losses.avg}) #, "Loss_Lp": lp_losses.avg, "Loss_Lq": lq_losses.avg

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
    
    def load_checkpoint(self, folder, filename):
        """
        Loads model from a file.
        :param folder:          source folder
        :param filename:        filename
        """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.load_state_dict(checkpoint['state_dict'])
        # self.load_state_dict(torch.load(filepath))
    
    def save_checkpoint(self, folder, filename):
        """
        Saves model to a file.
        :param folder:          destination folder
        :param filename:        filename
        """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.state_dict(),
        }, filepath)


    def load_first_checkpoint(self, folder, iteration):
        """
        loads model from a file, only used when loading a model at the start of the program
        :param folder:          source folder
        :param iteration:       iteration number from model
        """
        filename = "checkpoint_" + str(iteration) + ".h5"
        filepath = os.path.join(folder, filename)
        map_location = None if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.load_state_dict(checkpoint['state_dict'])
        # self.load_state_dict(torch.load(filepath))




# # Save
# torch.save(nnet.state_dict(), 'model.pth')

# # Load
# nnet.load_state_dict(torch.load('model.pth'))
# nnet.eval()  # Set to evaluation mode
