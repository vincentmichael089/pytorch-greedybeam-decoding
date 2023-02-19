import torch
import torch.nn as nn
import heapq
import numpy as np

BOS_IDX, EOS_IDX, PAD_IDX = 0,1,2 # change this 

class HeapItem:
  def __init__(self, p, t):
    self.p = p
    self.t = t

  def __lt__(self, other):
    return self.p < other.p


def create_mask(input_sequence, pad_symbol):
      cross_attn = input_sequence
      cross_attn_padding_mask = (cross_attn == PAD_IDX)
      char_padding_mask = (input_sequence == PAD_IDX)

      return cross_attn_padding_mask, char_padding_mask

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).cuda()
    return mask

def greedy_decode(model, input_sequence, max_len, start_symbol, pad_symbol):
  cross_attn_padding_mask, char_padding_mask = create_mask(input_sequence, pad_symbol)  
  memory = model.encode(input_sequence, char_padding_mask)
  yields = torch.ones(1, memory.shape[0]).T.fill_(start_symbol).type(torch.long).cuda()

  for _ in range(max_len-1):
    memory = memory.cuda()
    tgt_mask = (generate_square_subsequent_mask(yields.size(1)).type(torch.bool)).cuda()
    out = model.decode(yields, memory, tgt_mask, None, cross_attn_padding_mask)
    prob = nn.functional.log_softmax(model.linear(out[:, -1]), dim = 1)
    _, next_word = torch.max(prob, dim=1)

    yields = torch.cat([yields, next_word.unsqueeze(dim=0).T], dim=1)
  return yields

def beam_decode(model, input_sequence, max_len, start_symbol, end_symbol, pad_symbol, beam_width):
  batch_size = input_sequence.size(0)
  candidates = []
  cross_attn_padding_mask, char_padding_mask = create_mask(input_sequence, pad_symbol)
  memory = model.encode(input_sequence, char_padding_mask)

  for item in range(batch_size):
    heap = []
    yields = torch.ones(1, memory[item].unsqueeze(0).shape[0]).T.fill_(start_symbol).type(torch.long)
    tgt_mask = (generate_square_subsequent_mask(yields.size(1)).type(torch.bool)).cuda()
    out = model.decode(yields, memory[item].unsqueeze(0), tgt_mask, None, cross_attn_padding_mask[item].unsqueeze(0)) 
    log = nn.functional.log_softmax(model.linear(out[:, -1]), dim = 1)
    top_k_log, top_k_indexes = torch.topk(log, beam_width)

    for i, v in enumerate(top_k_indexes[0]):
      heapq.heappush(heap, HeapItem(top_k_log[0][i].item(), torch.cat([yields, v.unsqueeze(0).unsqueeze(0)], dim=1)))
    
    for j in range(max_len-1):
      heaptemp = heapq.nlargest(beam_width, heap)
      heap = []
      tgt_mask = (generate_square_subsequent_mask(j+2).type(torch.bool)).cuda()
      for tem in heaptemp:
        current_log = tem.p
        current_generation = tem.t
        if current_generation.squeeze().tolist()[-1] == end_symbol:
          heapq.heappush(heap, HeapItem(current_log, current_generation))
          continue
        else:
          out = model.decode(current_generation, memory[item].unsqueeze(0), tgt_mask, None, cross_attn_padding_mask[item].unsqueeze(0)) 
          log = nn.functional.log_softmax(model.linear(out[:, -1]), dim = 1)
          current_top_k_log, current_top_k_indexes = torch.topk(log, beam_width)

          for i, v in enumerate(current_top_k_indexes[0]):
            score = (current_log * np.power((j+1), 1) + current_top_k_log[0][i].item())/ np.power((j+2), 1) 
            prede = torch.cat([current_generation, v.unsqueeze(0).unsqueeze(0)], dim=1)
            try:
              heapq.heappush(heap, HeapItem(score, prede))
            except RuntimeError:
              print(heap)
              print(score)
              print(prede)
            
    candidate = heapq.nlargest(1, heap) #
    if candidate:
      candidates.append([cand.t.squeeze() for cand in candidate])

  return candidates
  
def translate(model, input_sequence, type, beam_width):
  model.eval()
  if type == "greedy":
    return greedy_decode(model, input_sequence, max_len=150, start_symbol=BOS_IDX, pad_symbol=PAD_IDX)
  elif type == "beam":
    return beam_decode(model, input_sequence, max_len=150, start_symbol=BOS_IDX, end_symbol=EOS_IDX, pad_symbol=PAD_IDX, beam_width=beam_width)

