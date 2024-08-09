import warnings

warnings.filterwarnings('ignore')
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1
import torch
from torch.utils.tensorboard import SummaryWriter
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class QURNNCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, cell_dim, bias, activation):
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
        self.bias = bias

        self.W_time = nn.Linear(input_dim + hidden_dim, cell_dim, bias)
        self.W_absorb = nn.Linear(input_dim + hidden_dim, cell_dim, bias)
        # self.up_h1 = nn.Linear(input_dim + hidden_dim, hidden_dim, bias)
        # self.up_h2 = nn.Linear(input_dim + cell_dim + hidden_dim, hidden_dim, bias)

        self.energy_h = nn.Linear(hidden_dim + input_dim, hidden_dim, bias)
        self.h_weight = nn.Linear(hidden_dim + input_dim, hidden_dim, bias)
        # self.state_linear = nn.Linear(cell_dim ,hidden_dim, bias)
        self.state_linear = nn.Linear(cell_dim + input_dim, hidden_dim,bias=False)
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

    def forward(self, input_tensor, cur_state):

        h_cur, excited_cell, time_cell = cur_state
        
        # initial
        # time_cell = time_cell * 0 
        # excited_cell = excited_cell * 0
        # from IPython import embed
        # embed()

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        
        # energy_init = self.sigmoid(self.energy_init(input_tensor))
        # ratio = self.sigmoid(self.time_ratio(input_tensor))
        # # Count the leap neuron
        
        up_time = self.tanh(self.W_time(combined))
        # time_cell = ratio * (time_cell+ up_time) - 1
        time_cell = self.relu(time_cell+ up_time) - 1
        leap = time_cell <= 0

        time_cell = (1-leap.type(torch.float32)) * time_cell
        

        # update excited state (release)
        energy = leap.type(torch.float32) * excited_cell
        # excited_cell = excited_cell - energy +  energy_init * (1-leap.type(torch.float32))
        excited_cell = excited_cell - energy

        # update excited state (absorb)
        absorb_energy = self.sigmoid(self.W_absorb(combined))
        excited_cell = self.relu(excited_cell + absorb_energy)      
        
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
class Model(nn.Module):
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

    def __init__(self, configs, batch_first=True, bias=True, return_all_layers=False):
        super(Model, self).__init__()
        # self._check_kernel_size_consistency(kernel_size)
        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        # kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        self.input_dim =configs.d_model
        self.hidden_dim = eval(configs.hidden_dim)  # [128] [64,128,256,512,256,128,64]
        self.cell_dim = eval(configs.cell_dim)  # [64] [64,128,256,512,256,128,64]
        self.num_layers = len(self.hidden_dim)
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.batch_first = batch_first
        self.output_dim = configs.d_model
        self.bias = bias
        self.return_all_layers = return_all_layers
        # self.relu = nn.ReLU()
        if configs.activation == 'relu':
            self.activation = nn.ReLU()
        elif configs.activation == 'gelu':
            self.activation = nn.GELU()
        elif configs.activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif configs.activation == 'tanh':
            self.activation = nn.Tanh()

        self.task_name = configs.task_name
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(self.output_dim, configs.c_out, bias=True)
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,configs.dropout)
        # self.input_linear = nn.Linear(configs.enc_in,configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.relu = nn.ReLU()

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(QURNNCell(input_dim=cur_input_dim,
                                       hidden_dim=self.hidden_dim[i],
                                       cell_dim=self.cell_dim[i],
                                       bias=self.bias,
                                       activation=configs.activation))

        self.cell_list = nn.ModuleList(cell_list)

        self.output_weight = nn.Sequential(nn.Linear(self.hidden_dim[-1], self.hidden_dim[-1]), 
                                            nn.LayerNorm(self.hidden_dim[-1]),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.hidden_dim[-1], self.output_dim),)

    def qurnnblock(self, input_tensor, hidden_state=None):
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
        # print("input_tensor:",input_tensor)
        if not self.batch_first:
            # (t, b, c) -> (b, t, c)
            input_tensor = input_tensor.permute(1, 0, 2)
        # from IPython import embed
        # embed()
        # print(input_tensor.shape)

        b, seq_len, c = input_tensor.size()
        # print(b, period, seq_len, c)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             cell_dim=self.cell_dim,
                                             hidden_dim=self.hidden_dim)

        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c_e, c_t = hidden_state[layer_idx]
            hidden = []
            for t in range(seq_len):
                h, c_e, c_t = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, ],
                                                        cur_state=[h, c_e, c_t])

                hidden.append(h)

            layer_output = torch.stack(hidden, dim=1)
            cur_layer_input = layer_output

        output = self.activation(self.output_weight(layer_output))

        # print("output:",output)
        return hidden, output

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

    def forward(self, x_enc, x_mask_enc, x_dec, x_mask_dec, mask=None):
        if self.task_name == 'classification':
            pass
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            result = self.forcasting(x_enc, x_mask_enc)
            return result[:, -self.pred_len:, :]


    def forcasting(self, x_enc, x_mask_enc, mask=None):
        
        # print(x_enc)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, unbiased=False, keepdim=True))
        x_enc = x_enc / stdev
        # print(x_enc.shape)
        # has_nan = torch.isnan(x_enc).any().item()
        # print("has nan：",has_nan)

        enc_out = self.enc_embedding(x_enc, x_mask_enc)  # [B,T,C]
        
        # enc_out = self.relu(self.input_linear(x_enc))
        # print(enc_out.shape)
        # has_nan = torch.isnan(enc_out).any().item()
        # print("has nan：",has_nan)
        # print(enc_out)
        # enc_out,_ = self.attention(enc_out_embed,enc_out_embed,enc_out_embed)


        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        # enc_out_1, (weight1, weight2) = self.slice(enc_out, 2)
        # print(enc_out.shape) batchsize 144,16
        # print(enc_out)
        _, out = self.qurnnblock(enc_out)  # input : b, seq_len, c
        # out = weight1 * out[:, 0, :, :].squeeze(dim=1) + weight2 * out[:, 1, :, :].squeeze(dim=1)
        # torch.Size([32, 192, 128])

        out = self.activation(out)

        # residual
        out = torch.add(out, enc_out)
        out = self.activation(out)

        # projection back
        out = self.projection(out)
        # torch.Size([32, 192, 7])

        # out = self.activation(out)

        # de-nomarlization from non-stationary transformer

        out = out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        # print(means)
        out = out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))

        return out

    def classitication(self, x_enc, x_mark_enc):
        print("x_enc shape ", x_enc.shape)
        print("x_mark_enc shape ", x_mark_enc)
        enc_out = self.enc_embedding(x_enc)
        print("embed output shape:", enc_out.shape)
        enc_out = self.activation(enc_out)
        output = self.dropout(enc_out)

        output = self.classi_linear(output)
        print("linear output shape:", output)
        output = self.qurnnblock(output)[2]

        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output
