import torch

def train(model, loaders, optimizer, writer, n_epochs, ckpt_path, device='cpu'):
  epoch_losses = []
  def run_epoch(train_or_eval):
    epoch_loss = 0.
    epoch_auc = 0.
    
    epoch_logits = []
    epoch_correctness = []
    for i, batch in enumerate(loaders[train_or_eval], 1):
      in_data, seq_len, next_skills, correctness = batch
      in_data, seq_len, next_skills, correctness = in_data.to(device), seq_len.to(device), next_skills.to(device), correctness.to(device)
      
      if train_or_eval == 'train':
        model.train() # train mode for dropout and batch normalization
        optimizer.zero_grad()
      else:
        model.eval() # eval mode for dropout and batch normalization
        
      logits = model(in_data, seq_len, next_skills)
      mask = model.selecting_mask(seq_len, in_data.size(1))
      batch_loss = model.loss(logits, correctness, mask)
      epoch_loss += batch_loss.item()
      
      if train_or_eval == 'train':
        batch_loss.backward()
        optimizer.step()
        
      epoch_logits.append(logits.masked_select(mask).detach().cpu())
      epoch_correctness.append(correctness.masked_select(mask).detach().cpu())
    
    epoch_loss /= i
    epoch_auc = model.auc(torch.cat(epoch_logits), torch.cat(epoch_correctness))
    
    losses[train_or_eval] = epoch_loss
    aucs[train_or_eval] = epoch_auc
    
    if writer is None:
      print('epoch %d %s loss %.4f auc %.4f' % (epoch, train_or_eval, epoch_loss, epoch_auc))
    elif train_or_eval == 'eval':
      writer.add_scalars('loss', 
                         tag_scalar_dict={'train': losses['train'], 
                                          'eval': losses['eval']}, 
                         global_step=epoch)
      writer.add_scalars('auc', 
                         tag_scalar_dict={'train': aucs['train'], 
                                          'eval': aucs['eval']}, 
                         global_step=epoch)
        
  # main statements
  losses = {}
  aucs = {}
  
  for epoch in range(1, n_epochs+1):
    run_epoch('train')
    run_epoch('eval')
    
    # For instructional purpose, show how to save checkpoints
    if ckpt_path is not None:
      torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'losses': losses,
        'aucs': aucs,
      }, '%s/%d.pt' % (ckpt_path, epoch))