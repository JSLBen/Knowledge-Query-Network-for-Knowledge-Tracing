from data_util import get_csv_fname, read_csv, get_num_skills
from train_util import train
from dataset import KQNDataset, PadSequence
from model import KQN
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from tensorboardX import SummaryWriter
from torch.optim import Adam
import torch
import os

def kqn_main():
  # argument parser
  # default hyperparams not set to optimal
  # dropout not used in the implementation
  parser = ArgumentParser()
  parser.add_argument('--dataset', type=str, default='assist0910', help='choose from assist0910, assist15, statics11, and synthetic-5')
  parser.add_argument('--version', type=int, default=None, help='if dataset==synthetic-5, choose from 0 to 19')
  parser.add_argument('--min_seq_len', type=int, default=2, help='minimum threshold of number of time steps to discard student problem-solving records.')
  parser.add_argument('--rnn', type=str, default='lstm', help='rnn type. one of lstm and gru.')
  parser.add_argument('--hidden', type=int, default=128, help='dimensionality of skill and knowledge state vectors')
  parser.add_argument('--rnn_hidden', type=int, default=128, help='number of hidden units for knowledge state encoder rnn')
  parser.add_argument('--mlp_hidden', type=int, default=128, help='number of hidden units for skill encoder mlp')
  parser.add_argument('--layer', type=int, default=1, help='number of rnn layers')
  parser.add_argument('--gpu', type=int, default=-1, help='which gpu to use. default to -1: not using any')
  parser.add_argument('--lr', type=float, default=0.001, help='learning rate for adam')
  parser.add_argument('--batch', type=int, default=100, help='batch size')
  parser.add_argument('--ckpt', type=str, default='./ckpt', help='default checkpoint path')
  parser.add_argument('--epoch', type=int, default=100, help='number of epochs')
  parser.add_argument('--optim', type=str, default='adam', help='optimizer to use. currently only adam is implemented.')
  
  args = parser.parse_args()
  dataset = args.dataset
  version = args.version
  min_seq_len = args.min_seq_len
  rnn_type = args.rnn
  n_hidden = args.hidden
  n_rnn_hidden = args.rnn_hidden
  n_mlp_hidden = args.mlp_hidden
  n_rnn_layers = args.layer
  gpu = args.gpu
  lr = args.lr
  batch_size = args.batch
  ckpt_path = args.ckpt
  n_epochs = args.epoch
  opt_str = args.optim
  
  if ckpt_path is not None:
    if not(os.path.exists(ckpt_path)):
      os.makedirs(ckpt_path)

  if gpu == -1:
    DEVICE = 'cpu'
  elif torch.cuda.is_available():
    DEVICE = gpu
    
  # load data
  n_skills = get_num_skills(dataset)
  fnames = {'train': get_csv_fname(True, dataset, version), 'eval': get_csv_fname(False, dataset, version)}
  datasets = {'train': read_csv(fnames['train'], min_seq_len), 'eval': read_csv(fnames['eval'])}
  datasets = {'train': KQNDataset(datasets['train'][0], datasets['train'][1], datasets['train'][2], n_skills), 
              'eval': KQNDataset(datasets['eval'][0], datasets['eval'][1], datasets['eval'][2], n_skills)}
  dataloaders = {'train': DataLoader(datasets['train'], batch_size=batch_size, drop_last=False, collate_fn=PadSequence(), shuffle=True), 
                 'eval': DataLoader(datasets['eval'], batch_size=batch_size, drop_last=False, collate_fn=PadSequence())}
    
  model = KQN(n_skills, n_hidden, n_rnn_hidden, n_mlp_hidden, n_rnn_layers, rnn_type, DEVICE).to(DEVICE)
  
  if opt_str == 'adam': opt_class = Adam
  
  optimizer = opt_class(model.parameters(), lr=lr)
  writer = SummaryWriter('./logs')
  
  train(model, dataloaders, optimizer, writer, n_epochs, ckpt_path, DEVICE)
  
if __name__=='__main__':
  kqn_main()