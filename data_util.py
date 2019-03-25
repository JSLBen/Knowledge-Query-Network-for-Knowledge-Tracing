from typing import List
data_folder = './data'

# get the csv file name
# dataset is in  ['assist0910', 'assist15', 'statics11', 'synthetic-5']
# if dataset=='synthetic-5', version should be specified from 0 to 19
def get_csv_fname(train: bool, dataset: str, version: int=None) -> str:
  fname = 'train.csv' if train else 'test.csv'
  if dataset=='synthetic-5': fname = 'naive_c5_q50_s4000_v%d_%s' % (version, fname)
  return '%s/%s/%s' % (data_folder, dataset, fname)

def get_num_skills(dataset: str) -> int:
  if dataset=='assist0910': return 110
  elif dataset=='assist15': return 100
  elif dataset=='statics11': return 1223
  elif dataset=='synthetic-5': return 50
  else: raise NotImplementedError('Invalid Dataset')
    
# fname: csv file name for a dataset
# minimum_seq_len: minimum sequence length required. default to 2
def read_csv(fname: str, minimum_seq_len: int=2) -> (List[int], List[List[int]], List[List[int]]):
  with open(fname, 'r') as f:
    data = f.read()
  
  data = data.split('\n')
  
  # remove all white spaces at both ends
  while data[0] == '':
    data = data[1:]
    
  while data[-1] == '':
    data = data[:-1]
    
  seq_len = []
  skill_ids = []
  correctness = []
  i = 0
  while i < len(data):
    line = data[i]
    if i % 3 == 0: # seq_len
      if int(line) >= minimum_seq_len: seq_len.append(int(line))
      else:
        i += 3
        continue
    elif i % 3 == 1: # skill ids
      line = line.split(',')
      skill_ids.append([int(e)-1 for e in line if e != '']) # -1 since id starts from 1
    else: # correctness
      line = line.split(',')
      correctness.append([int(e) for e in line if e != ''])
      
      # for integrity, check the lengths
      assert seq_len[-1] == len(skill_ids[-1]) == len(correctness[-1])
      
    i+= 1
      
  return seq_len, skill_ids, correctness