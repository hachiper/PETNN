import torch as t
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging
import torch.optim as optim
import os
import jieba
import gensim
from gensim.test.utils import  datapath,get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
import argparse
import re
import time
# from cell import QURNN
from qurnn_ori import QURNN
import torch
import thop
#from qn_rnn import *
# from allennlp.commands.elmo import ElmoEmbedder
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("./log")




class textCNN(nn.Module):
    # 多通道textcnn
    def __init__(self, args,vectors=None):
        super(textCNN,self).__init__()
        self.args=args

        self.label_num=args.label_num   #标签个数
        self.filter_num=args.filter_num #卷积核个数
        self.filter_sizes=[int(fsz) for fsz in args.filter_sizes]
        self.vocab_size=args.vocab_size
        self.embedding_dim=args.embedding_dim

        self.embedding=nn.Embedding(self.vocab_size,self.embedding_dim)
        # # requires_grad指定是否在训练过程中对词向量的权重进行微调
        self.embedding.weight.requires_grad = True
        if args.static:  # 如果使用预训练词向量，则提前加载，当不需要微调时设置freeze为True
            self.embedding = self.embedding.from_pretrained(vectors, freeze=not args.fine_tune)
        channel_in=1
        # nn.ModuleList相当于一个卷积的列表，相当于一个list
        # 卷积核宽度与embeding-dim相同，相当于一维卷积
        # nn.Conv1d()是一维卷积。in_channels：词向量的维度， out_channels：输出通道数
        # nn.MaxPool1d()是最大池化，此处对每一个向量取最大值，所有kernel_size为卷积操作之后的向量维度
        self.convs = nn.ModuleList([nn.Sequential(
             nn.Conv2d(channel_in, self.filter_num, (kernel, self.embedding_dim)),
             nn.ReLU(),# 经过卷积之后，得到一个维度为sentence_max_size - kernel + 1的一维向量
             nn.MaxPool2d((args.sentence_max_size - kernel + 1, 1))
         )
             for kernel in self.filter_sizes])
        # self.convs2 = nn.ModuleList([nn.Sequential(
        #     nn.Conv2d(channel_in, self.filter_num, (kernel, self.embedding_dim)),
        #     nn.ReLU(),  # 经过卷积之后，得到一个维度为sentence_max_size - kernel + 1的一维向量
            # nn.MaxPool2d((args.sentence_max_size - kernel + 1, 1))
        # )
        #     for kernel in self.filter_sizes])

        # self.convs = nn.ModuleList([
        #     nn.Conv2d(channel_in, self.filter_num, (fsz, self.embedding_dim))
        #      for fsz in self.filter_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(self.filter_sizes) * self.filter_num, self.label_num)
        self.qurnn = QURNN(input_dim=args.embedding_dim, cell_dim=args.embedding_dim,hidden_dim=args.embedding_dim,output_dim=self.label_num)
        # self.qurnn = QURNN(input_dim=args.embedding_dim, cell_dim=args.embedding_dim,hidden_dim=args.embedding_dim,output_dim=128)

        # self.qurnn_forward = QURNN(input_dim=args.embedding_dim, cell_dim=args.embedding_dim,hidden_dim=args.embedding_dim,output_dim=self.label_num)
        # self.qurnn_back = QURNN(input_dim=args.embedding_dim, cell_dim=args.embedding_dim,hidden_dim=args.embedding_dim,output_dim=self.label_num)
        # self.output_linear = nn.Linear(self.label_num*2,self.label_num)
        # self.rnn_fc = nn.Linear(128,self.label_num)
        # self.lstm = lstm_test(self.embedding_dim, self.label_num)
        # self.gru = gru_test(self.embedding_dim, self.label_num)
        # self.linear = nn.Linear(args.embedding_dim,self.label_num)
        # self.fc_1 = nn.Linear(args.embedding_dim * args.sentence_max_size,128)
        # self.fc_2 = nn.Linear(128,10)
        self.sigmoid = nn.Sigmoid()
    
    def forward_text(self,x):
        # Conv2d的输入是个四维的tensor，每一位分别代表batch_size、channel、length、width
        in_size = x.size(0)  # x.size(0)，表示的是输入x的batch_size
        out = [conv(x) for conv in self.convs]
        # print(out.shape)
        out = t.cat(out, dim=1)
        # print(out.shape)
        out = out.view(in_size, -1)  # 设经过max pooling之后，有output_num个数，将out变成(batch_size,output_num)，-1表示自适应
        # print(out.shape)
        out = F.dropout(out)
        out = self.fc(out)  # nn.Linear接收的参数类型是二维的tensor(batch_size,output_num),一批有多少数据，就有多少行
        return out
    
    def forward(self,x):
        # 89%
        # 88.532 %
        # Conv2d的输入是个四维的tensor，每一位分别代表batch_size、channel、length、width
        # _,out = self.qurnn(x)
        # print(out.shape)
        # out = self.rnn_fc(out[:,-1,:])
        _,_,out = self.qurnn(x)
        return out
    def forward_lstm(self,x):
        x = x.squeeze(dim=1)
        out = self.lstm(x)
        # out = self.gru(x)
        # print(out.shape)
        return out
    
    def forward_fc(self,x):
        embed = x.view(x.size(0), -1)
        out = self.fc_1(embed)
        out = F.relu(out)
        out = self.fc_2(out)
        return out

class LSTM_TEST(nn.Module):
    def __init__(self, input_dim, ouput_size):
        super(LSTM_TEST, self).__init__()
        self.lstm = nn.LSTM(input_dim, 256, batch_first=True)
        self.weight = nn.Linear(256, ouput_size)
        self.drop = nn.Dropout(0.8)

    def forward(self, input_x):
        (y, (h_n, c_n)) = self.lstm(input_x)
        y = self.drop(y)
        out = self.weight(y[:, -1, :]).squeeze(dim=1)
        return out
    
def lstm_test(input_dim, output_dim):
    net = LSTM_TEST(input_dim, output_dim)
    return net

class GRU_TEST(nn.Module):
    def __init__(self, input_dim, output_size):
        super(GRU_TEST, self).__init__()
        self.gru = nn.GRU(input_dim, 64, batch_first=True)
        self.fc1 = nn.Linear(64, output_size)

    def forward(self, input_x):
        y, hn = self.gru(input_x)
        # print(y.shape)
        out = self.fc1(y[:, -1, :]).squeeze(dim=1)
        # out = self.fc1(y)
        return out
def gru_test(input_dim, out_dim):
    net = GRU_TEST(input_dim, out_dim)
    return net





# [-0.01, -0.0001,……]



import math 

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.lr * (0.5 ** ((epoch - 1) // 2))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.lr /2 * (1 + math.cos(epoch / args.epoch * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class MyDataset(Dataset):
    """MyDataset的实现原理就是通过遍历file_list，得到每一个文件路径名，根据路径名，将其内容读到内存中，
    通过generate_tensor()函数将文件内容转化为tensor，函数返回tensor与对应的label，其中index就是list的下标"""

    def __init__(self, file_list, label_list, sentence_max_size, embedding, word2id, stopwords=None):
        self.x = file_list
        self.y = label_list
        self.sentence_max_size = sentence_max_size
        self.embedding = embedding
        self.word2id = word2id
        self.stopwords = stopwords

    def __getitem__(self, index):
        # 读取文章内容
        words = []
        with open(self.x[index], "r", encoding="utf8") as file:
            for line in file.readlines():
                words.extend(segment(line.strip()))
        # 生成文章的词向量矩阵
        tensor = generate_tensor(words, self.sentence_max_size, self.embedding, self.word2id)
        return tensor, self.y[index]

    def __len__(self):
        return len(self.x)


def get_file_list(source_dir):
    file_list = []  # 文件路径名列表
    # os.walk()遍历给定目录下的所有子目录，每个walk是三元组(root,dirs,files)
    # root 所指的是当前正在遍历的这个文件夹的本身的地址
    # dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
    # files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
    # 遍历所有评论
    for root, dirs, files in os.walk(source_dir):
        # from IPython import embed
        # embed()
        if root == './aclImdb/train/unsup' or root == './aclImdb/train' or root == './aclImdb/train\\unsup':
            continue
        if root == './aclImdb/test/unsup' or root == './aclImdb/test' or root == './aclImdb/test\\unsup':
            continue
        file = [os.path.join(root, filename) for filename in files]
        # print(root,dir,file)    #file和
        file_list.extend(file)
    # print('len of filelist:',len(file_list),file_list)
    return file_list


def get_label_list(file_list):
    # print(len(file_list))
    # 提取出标签名
    # print(file_list)
    label_name_list = [file.split("\\")[1] for file in file_list]
    # 标签名对应的数字
    label_list = []
    for label_name in label_name_list:
        if label_name == "neg":
            label_list.append(0)
        elif label_name == "pos":
            label_list.append(1)
    return label_list

def segment(content):
    # regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')
    # text = regex.sub(' ', content)
    text = re.sub('[^\w ]', '', content)
    return [word for word in jieba.cut(text) if word.strip()]


'''先将一篇评论进行分词，然后将每个词转换为对应的词向量。最终每篇评论，会变成[sentence_max_size,vec_dim]的矩阵'''
def generate_tensor(sentence, sentence_max_size, embedding, word2id):
    """
    对一篇文章生成对应的词向量矩阵
    :param sentence:一篇文章的分词列表
    :param sentence_max_size:认为设定的一篇文章的最大分词数量
    :param embedding:词向量对象
    :param word2id:字典{word:id}
    :return:一篇文章的词向量矩阵
    """
    tensor = t.zeros([sentence_max_size, embedding.embedding_dim])
    for index in range(0, sentence_max_size):
        if index >= len(sentence):
            vector = embedding.weight[word2id['0']]
            tensor[index] = vector
        else:
            word = sentence[index]
            if word in word2id:
                vector = embedding.weight[word2id[word]]
                tensor[index] = vector
            elif word.lower() in word2id:
                vector = embedding.weight[word2id[word.lower()]]
                tensor[index] = vector
    return tensor.unsqueeze(0)  # tensor是二维的，必须扩充为三维，否则会报错

from torch.optim.lr_scheduler import LambdaLR

def train_textcnn_model(net, train_loader, epoch, lr, args):
    print("begin training")
    if args.cuda:
        net.cuda()
    net.train()  # 必备，将模型设置为训练模式
    optimizer = optim.Adam(net.parameters(), lr=lr)
    #optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # criterion = nn.NLLLoss()
    # criterion = nn.MSELoss()
    for i in range(epoch):  # 多批次循环
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data,target=data.cuda(),target.cuda()

            optimizer.zero_grad()  # 清除所有优化的梯度

            output = net(data)  # 传入数据并前向传播获取输出
            # target = F.one_hot(target, num_classes=2).float()
            loss = criterion(output, target)
            iteration = (epoch -1)*len(train_dataloader) + batch_idx
            writer.add_scalar("train/loss", loss, iteration)
            loss.backward()
            optimizer.step()

            # 打印状态信息
            logging.info("train epoch=" + str(i) + ",batch_id=" + str(batch_idx) + ",loss=" + str(loss.item() / 64))
        net.eval()
        textcnn_model_test(net,test_dataloader,args=parse)
        net.train()
    print('Finished Training')

def textcnn_model_test(net, test_loader,args):
    if args.cuda:
        net.cuda()
    net.eval()  # 必备，将模型设置为训练模式
    correct = 0
    total = 0
    test_acc = 0.0
    with t.no_grad():
        for i, (data, label) in enumerate(test_loader):
            logging.info("test batch_id=" + str(i))
            # data=data.long()
            if args.cuda:
                data, label = data.cuda(), label.cuda()
            # data = data.squeeze(dim=1)
            #
            outputs = net(data)
            # torch.max()[0]表示最大值的值，troch.max()[1]表示回最大值的每个索引
            _, predicted = t.max(outputs.data, 1)  # 每个output是一行n列的数据，取一行中最大的值
            total += label.size(0)
            correct += (predicted == label).sum().item()
            logging.info('Accuracy of the network on test set: %.3f %%' % (100 * correct / total))
        writer.add_scalar("vali/accuracy", 100 * correct / total)
            # test_acc += accuracy_score(torch.argmax(outputs.data, dim=1), label)
            # logging.info("test_acc=" + str(test_acc))

def transfer(glove_dir,word2vec_dir):

    glove_input_file=datapath(glove_dir)
    word2vec_output_file=get_tmpfile(word2vec_dir)   #创建临时文件
    (count, dimensions) = glove2word2vec(glove_input_file, word2vec_output_file)
    # print(count, '\n', dimensions)
    return count, dimensions


def parse():
    parser = argparse.ArgumentParser(description='TextCNN text classifier')
    parser.add_argument('-vocab-size', type=int, default=89527, help='评论词表大小')
    parser.add_argument('-lr', type=float, default=0.001, help='学习率')
    parser.add_argument('-batch-size', type=int, default=64)
    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-filter-num', type=int, default=100, help='卷积核的个数')
    parser.add_argument('-filter-sizes', type=str, default=[2,3,4], help='不同卷积核大小')
    parser.add_argument('-embedding-dim', type=int, default=300, help='词向量的维度')
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-label_num', type=int, default=2, help='标签个数')
    parser.add_argument('-static', type=bool, default=True, help='是否使用预训练词向量')
    parser.add_argument('-fine-tune', type=bool, default=True, help='预训练词向量是否要微调')
    parser.add_argument('-sentence_max_size',type=int,default=300,help='评论的最大长度')
    parser.add_argument('-cuda', type=bool, default=True)
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--cell_dim', type=list, default=[64],help = 'dimension of cell')
    parser.add_argument('--hidden_dim', type=list, default=[64], help='dimension of hidden')
    parser.add_argument('--activation', type=str, default="gelu", help='activation')
    parser.add_argument('--dropout', type=float, default=0.8, help='drop')

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    # t.cuda.synchronize()
    start_time=time.time( )
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
    train_dir = './aclImdb/train'  # 训练集路径
    test_dir = "./aclImdb/test"  # 测试集路径
    # net_dir = "./model/net.pkl"
    params_dir="./model/gpu_params_300.pkl"
    parse=parse()


    #最好写绝对路径
    glove_dir = 'C:\learning\QURNN\glove6B\glove.6B.'+str(parse.embedding_dim)+'d.txt'
    #word2vec_dir = '/disk_tmp_2/haohy/ajy_project/glove6B/glove.6B.word2vec.'+str(parse.embedding_dim)+'d.txt'
    word2vec_dir = 'C:\learning\QURNN\glove6B\glove.6B.word2vec.'+str(parse.embedding_dim)+'d.txt'
    # print('count, dimensions',transfer(glove_dir,word2vec_dir))
      
        
    # 加载词向量模型
    logging.info("加载词向量模型")
    # 使用gensim载入word2vec词向量

    wvmodel = gensim.models.KeyedVectors.load_word2vec_format('./glove6B/glove.6B.word2vec.'+str(parse.embedding_dim)+'d.txt',
                                                        binary=False, encoding='utf-8')


    word2id = {}  # word2id是一个字典，存储{word:id}的映射
    for i, word in enumerate(wvmodel.index_to_key):
        word2id[word] = i
    # 根据已经训练好的词向量模型，生成Embedding对象

    embedding = nn.Embedding.from_pretrained(t.FloatTensor(wvmodel.vectors))
    # requires_grad指定是否在训练过程中对词向量的权重进行微调
    # embedding.weight.requires_grad = True

    # 获取训练数据
    logging.info("获取训练数据")
    train_filelist = get_file_list(train_dir)
    train_labellist = get_label_list(train_filelist)
    train_dataset = MyDataset(train_filelist, train_labellist, parse.sentence_max_size, embedding, word2id)
    train_dataloader = DataLoader(train_dataset, batch_size=parse.batch_size, shuffle=True)


    # 获取测试数据
    logging.info("获取测试数据")
    test_set = get_file_list(test_dir)
    test_label = get_label_list(test_set)
    test_dataset = MyDataset(test_set, test_label, parse.sentence_max_size, embedding, word2id)
    test_dataloader = DataLoader(test_dataset, batch_size=parse.batch_size, shuffle=True)

    # 定义模型
    net = textCNN(parse,vectors=t.FloatTensor(wvmodel.vectors))

    # 训练
    logging.info("开始训练模型")
    # t.cuda.synchronize()
    start_train = time.time()
    train_textcnn_model(net, train_dataloader, parse.epoch, parse.lr, args=parse)
    t.save(net.state_dict(), params_dir)  # 保存模型参数
    # t.cuda.synchronize()
    end_train = time.time()
    logging.info('train time cost:%f s' % (end_train - start_train))
    # 测试
    net = textCNN(parse, vectors=t.FloatTensor(wvmodel.vectors))
    net.load_state_dict(t.load(params_dir))
    textcnn_model_test(net,test_dataloader,args=parse)
    # t.cuda.synchronize()
    end_test = time.time()
    logging.info('test time cost:%f s' % (end_test - end_train))
    logging.info('overall time cost:%f s' % (end_test - start_time))
    
