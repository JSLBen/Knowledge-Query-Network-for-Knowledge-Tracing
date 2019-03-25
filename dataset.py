from typing import List, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np
class KQNDataset(Dataset):
  def __init__(self, seq_len: List[int], skill_ids: List[List[int]], correctness: List[List[int]], n_skills: int):
    assert len(seq_len) == len(skill_ids) == len(correctness)
    self.seq_len = seq_len
    self.skill_ids = skill_ids
    self.correctness = correctness
    self.n_skills = n_skills
    self.two_eye = np.eye(2*n_skills) # helper variable for making a one-hot vector for rnn input
    self.eye = np.eye(n_skills) # helper variable for making a one-hot vector for skills
    
  def __len__(self):
    return len(self.seq_len)
  
  def __getitem__(self, index: int):
    seq_len = self.seq_len[index]
    skill_ids = np.array(self.skill_ids[index], dtype=np.int32) # (seq_len,)
    correctness = np.array(self.correctness[index], dtype=np.int32) # (seq_len,)
    
    in_data = self.two_eye[correctness[:-1]*self.n_skills+skill_ids[:-1]] # (seq_len-1, 2*self.n_skills)
    next_skills = self.eye[skill_ids[1:]] # (seq_len-1, self.n_skills)
    correctness = correctness[1:] # (seq_len-1,)
    return torch.FloatTensor(in_data), torch.LongTensor([seq_len-1]), torch.FloatTensor(next_skills), torch.FloatTensor(correctness)
  
class PadSequence(object):
  def __call__(self, batch: List[Tuple[torch.Tensor]]): # [(seq_len, 2*n_skills), (1,), (seq_len, n_skills), (seq_len,)] x batch_size
    # Sort the batch in the descending order
    batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    
    # Pad each data
    in_data = torch.nn.utils.rnn.pad_sequence([x[0] for x in batch], batch_first=True) # (batch_size, max_seq_len, 2*n_skills)
    next_skills = torch.nn.utils.rnn.pad_sequence([x[2] for x in batch], batch_first=True) # (batch_size, max_seq_len, n_skills)
    correctness = torch.nn.utils.rnn.pad_sequence([x[3] for x in batch], batch_first=True) # (batch_size, max_seq_len)
    
    # seq_len
    seq_len = torch.cat([x[1] for x in batch]) # (batch_size,)
    
    return in_data, seq_len, next_skills, correctness
  