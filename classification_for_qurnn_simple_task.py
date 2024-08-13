import torch as t
import torch.nn as nn
import torch.optim as optim

import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("./log")
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class QURNNCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, cell_dim, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(QURNNCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell_dim = cell_dim

        # self.kernel_size = kernel_size
        # self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
        #                       out_channels=4 * self.hidden_dim,
        #                       kernel_size=self.kernel_size,
        #                       padding=self.padding,
        #                       bias=self.bias)

        self.W_time = nn.Linear(input_dim + hidden_dim, cell_dim, bias)
        self.W_absorb = nn.Linear(input_dim + hidden_dim, cell_dim, bias)
        # self.up_h1 = nn.Linear(input_dim + hidden_dim, hidden_dim, bias)
        # self.up_h2 = nn.Linear(input_dim + cell_dim + hidden_dim, hidden_dim, bias)

        self.energy_h = nn.Linear(hidden_dim + input_dim, hidden_dim, bias)
        self.h_weight = nn.Linear(hidden_dim + input_dim, hidden_dim, bias)

        # self.cell_t = nn.Parameter(torch.rand((cell_dim)), requires_grad=True)
        # self.cell_h = nn.Parameter(torch.rand((cell_dim)), requires_grad=True)
        self.time_ratio = nn.Linear(input_dim, hidden_dim, bias)
        self.energy_init = nn.Linear(input_dim, hidden_dim, bias)
        self.time = nn.Linear (input_dim,hidden_dim,bias)
        # self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    # def qu(self, x):
    #     x = torch.floor(x)

    #     emery_pre = 1/(x*x)
    #     emery_sub = 1/(self.b_ih)
    #     return emery_pre - emery_sub

    # def 

    def forward(self, input_tensor, cur_state):

        h_cur, excited_cell, time_cell = cur_state
        
        # initial
        # excited_cell = excited_cell * 0
        # time_cell = time_cell * 0
        # from IPython import embed
        # embed()

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        energy_init = self.sigmoid(self.energy_init(input_tensor))
        ratio = self.sigmoid(self.time_ratio(input_tensor))
        # # Count the leap neuron
        
        up_time = self.tanh(self.W_time(combined))
        time_cell = ratio * self.relu(time_cell+ up_time) - 1
        # time_cell = time_cell - 1
        leap = time_cell <= 0

        time_cell = (1-leap.type(torch.float32)) * time_cell
        

        # update excited state (release)
        energy = leap.type(torch.float32) * excited_cell
        excited_cell = excited_cell - energy +  energy_init * (1-leap.type(torch.float32))
        

        # update excited state (absorb)
        absorb_energy = self.sigmoid(self.W_absorb(combined))
        excited_cell = excited_cell + absorb_energy      
        
        h_up = h_cur * energy

        
        h_state = torch.tanh(self.energy_h(torch.cat([input_tensor, h_up], dim=1)))

        h_weight = self.sigmoid(self.h_weight(combined))
        h_next = self.tanh((1 - h_weight) * h_cur + h_weight * h_state)
        return h_next, excited_cell, time_cell

    def init_hidden(self, batch_size, cell_dim, hidden_dim):
        return (torch.rand(batch_size, hidden_dim, device=self.W_time.weight.device), 
                torch.rand(batch_size, cell_dim, device=self.W_time.weight.device), 
                torch.rand(batch_size, cell_dim, device=self.W_time.weight.device))


# [-0.01, -0.0001,……]

class QURNN(nn.Module):
    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, cell_dim, output_dim, num_layers=1,
                 batch_first=True, bias=True, return_all_layers=False):
        super(QURNN, self).__init__()

        # self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        # kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        cell_dim = self._extend_for_multilayer(cell_dim, num_layers)
        num_layers = len(cell_dim)
        if not len(cell_dim) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell_dim = cell_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.relu = nn.ReLU()

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(QURNNCell(input_dim=cur_input_dim,
                                       hidden_dim=self.hidden_dim[i],
                                       cell_dim=self.cell_dim[i],
                                       bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)
        # self.encode_gru = nn.GRU(input_size=self.input_dim)
        # self.decode_gru = nn.GRU(input_size=self.hidden_dim[-1])

        self.output_weight = nn.Sequential(nn.Linear(hidden_dim[-1], hidden_dim[-1]),
                                           nn.BatchNorm1d(hidden_dim[-1]),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(hidden_dim[-1], output_dim))

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        # lmd = 0.8
        # input_tensor = input_tensor.squeeze()
        # print(input_tensor.shape)
        if not self.batch_first:
            # (t, b, c) -> (b, t, c)
            input_tensor = input_tensor.permute(1, 0, 2)

        # from IPython import embed
        # embed()
        b, seq_len, c = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             cell_dim=self.cell_dim,
                                             hidden_dim=self.hidden_dim)

        layer_output_list = []
        last_state_list = []

        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c_e, c_t = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):

                h, c_e, c_t = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, ],
                                                        cur_state=[h, c_e, c_t])

                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c_e, c_t])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        # output, hidden_state= self.decode_gru(self.relu(last_state_list[0][0]))
        
        # output = self.output_weight(layer_output)

        # output = self.output_weight(output)
        # print(output.shape)
        output = self.output_weight(last_state_list[-1][0])
        # print(output)
        return layer_output_list, last_state_list, output

    def _init_hidden(self, batch_size, cell_dim, hidden_dim):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, cell_dim[i], hidden_dim[i]))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

def qurnn_test(input_dim, hidden_dim, cell_dim, output_dim):
    net = QURNN(input_dim, hidden_dim, cell_dim, output_dim)
    return net

class QURNN_MNIST(nn.Module):
    # 2024-05-17 10:24:00,725:INFO: Accuracy of the network on test set: 98.620 %
    # 28*3
    def __init__(self, input_dim=28, hidden_dim=64, cell_dim=64, output_dim=10,num_layer=1):
        super(QURNN_MNIST, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell_dim = cell_dim
        self.output_dim = output_dim
        self.num_layer = num_layer
        self.ln = torch.nn.LayerNorm(input_dim)
        self.net1 = QURNN(self.input_dim, self.hidden_dim, self.cell_dim, self.output_dim,self.num_layer)
        self.net2 = QURNN(self.input_dim, self.hidden_dim, self.cell_dim, self.output_dim,self.num_layer)
        self.fc = nn.Linear(self.output_dim,self.input_dim)
        self.fc_reverse = nn.Linear(self.input_dim,self.output_dim)
        self.fc1=nn.Linear(self.input_dim * self.input_dim,self.input_dim * self.input_dim)
        self.fc2=nn.Linear(self.input_dim * self.input_dim, self.output_dim)
        # self.net3 = QURNN(self.input_dim, self.hidden_dim*3, self.cell_dim*3, self.output_dim,self.num_layer)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()

    def forward_qun(self, x):
        # 2024-05-15 20:30:37,094:INFO: Accuracy of the network on test set: 97.430 %
        
        batch_size = x.size()[0]
        x = x.view(batch_size,-1)

        x = x.unsqueeze(dim=2)
        # print(x.shape)
        _,_,out = self.net1(x)
        # out = self.fc(out)
        # out = self.ln(out)
        # out = self.relu(out)
        # out = self.net2(out)[2]

        # out = self.tanh(self.fc_reverse(out))
        # print(out.shape)
        return out
    def forward(self,x):
        x = x.squeeze(dim=1)
        batch_size = x.size()[0]
        x = x.view(batch_size,-1)
        # print(x.shape)
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        # out = self.sigmoid(out)
        return out



def train(net, train_loader, epoch, lr, cuda):
    print("begin training")
    if cuda:
        net.cuda()
    net.train()  # 必备，将模型设置为训练模式
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1))
    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()
    # criterion = nn.MSELoss()
    for i in range(epoch):  # 多批次循环
        for batch_idx, (data, target) in enumerate(train_loader):
            # data=data.long(); print(data.dtype)
            # data = data.long()
            # target = target.long()
            if cuda:
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()  # 清除所有优化的梯度

            # data = data.squeeze(dim=1)

            output = net(data)  # 传入数据并前向传播获取输出
            # print(output.shape)
            # print(target.shape)
            # target = F.one_hot(target, num_classes=2)
            loss = criterion(output, target)
            iteration = (epoch - 1) * len(train_loader) + batch_idx
            writer.add_scalar("train/loss", loss, iteration)
            loss.backward()
            optimizer.step()
            # scheduler.step()

            # 打印状态信息
            logging.info("train epoch=" + str(i) + ",batch_id=" + str(batch_idx) + ",loss=" + str(loss.item() / 64))
        # print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
        # writer.add_scalar("loss", loss.detach())
    print('Finished Training')


def test(net, test_loader, cuda):
    if cuda:
        net.cuda()
    net.eval()  # 必备，将模型设置为训练模式
    correct = 0
    total = 0
    test_acc = 0.0
    with t.no_grad():
        for i, (data, label) in enumerate(test_loader):
            logging.info("test batch_id=" + str(i))
            # data=data.long()
            if cuda:
                data, label = data.cuda(), label.cuda()
            # data = data.squeeze(dim=1)
            #
            outputs = net(data)
            # torch.max()[0]表示最大值的值，troch.max()[1]表示回最大值的每个索引
            _, predicted = t.max(outputs.data, 1)  # 每个output是一行n列的数据，取一行中最大的值
            total += label.size(0)
            correct += (predicted == label).sum().item()
            logging.info('Accuracy of the network on test set: %.3f %%' % (100 * correct / total))

import argparse
import time
import logging
from torch.utils.data import DataLoader
from torchvision import transforms as tfs
from torchvision.datasets import MNIST,FashionMNIST

if __name__ == "__main__":
    # t.cuda.synchronize()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    params_dir = "./model/mnist.pkl"

    start_time = time.time()
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)

    # 定义数据
    data_tf = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize([0.5], [0.5])  # 标准化
    ])

    train_set = MNIST('./data', train=True, transform=data_tf,download=True)
    # train_set = FashionMNIST('./data', train=True, transform=data_tf,download=True)
    test_set = MNIST('./data', train=False, transform=data_tf,download=True)
    # test_set = FashionMNIST('./data', train=True, transform=data_tf,download=True)

    train_data = DataLoader(train_set, 64, True, num_workers=4) # 64,1,28,28
    test_data = DataLoader(test_set, 64, False, num_workers=4) # 64,1,28,28

    # 定义模型
    # net = QURNN_MNIST(input_dim=28*28, hidden_dim=28*28, cell_dim=28*28, output_dim=10)
    net = QURNN_MNIST(input_dim=28, hidden_dim=16, cell_dim=16, output_dim=10)
    # 训练
    logging.info("开始训练模型")
    # t.cuda.synchronize()
    criterion = nn.CrossEntropyLoss()

    optimzier = torch.optim.Adadelta(net.parameters(), 0.001)
    # 开始训练

    start_train = time.time()
    train(net, train_data, epoch=10, lr=0.001, cuda=True)
    t.save(net.state_dict(), params_dir)  # 保存模型参数
    # t.cuda.synchronize()
    end_train = time.time()
    logging.info('train time cost:%f s' % (end_train - start_train))
    # 测试
    # net = QURNN_MNIST(input_dim=28*28, hidden_dim=28*28, cell_dim=28*28, output_dim=10)
    net = QURNN_MNIST(input_dim=28, hidden_dim=16,cell_dim=16, output_dim=10)
    net.load_state_dict(t.load(params_dir))
    test(net, test_data,cuda=True)
    # t.cuda.synchronize()
    start_time = time.time()
    end_test = time.time()
    logging.info('test time cost:%f s' % (end_test - end_train))
    logging.info('overall time cost:%f s' % (end_test - start_time))
