# from collections import deque
import numpy as np
import random
import torch
import pickle as pickle
from torch.autograd import Variable
class rpm(object):
    # replay memory
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.priorities = []
        
        self.index = 0
        
    def append(self, obj):
        if self.size() > self.buffer_size:
            print('buffer size larger than set value, trimming...')
            self.buffer = self.buffer[(self.size() - self.buffer_size):]
            self.priorities = self.priorities[(self.size() - self.buffer_size):]
        elif self.size() == self.buffer_size:
            self.buffer[self.index] = obj
            self.priorities[self.index] = max(self.priorities, default=1)
            self.index += 1
            self.index %= self.buffer_size
        else:
            self.buffer.append(obj)
            self.priorities.append(max(self.priorities, default=1))

    def size(self):
        return len(self.buffer)

    def get_probabilities(self, priority_scale):
        # scaled_priorities = np.array(self.priorities)
        # with torch.no_grad():
        scaled_priorities_torch = Variable(torch.Tensor(self.priorities))**priority_scale
        denom = torch.sum(scaled_priorities_torch)
        sampled_probabilities = scaled_priorities_torch/denom
        return sampled_probabilities
    
    def get_importance(self, probabilities):
        importance = 1/len(self.buffer) * 1/probabilities
        importance_normalized = importance/max(importance)
        return importance_normalized


    def sample_batch(self, batch_size, device, only_state=False, priority_scale = 0.7):
        if self.size() < batch_size:
            MINIBATCH_SIZE = self.size()
            # batch = random.sample(self.buffer, self.size())
        else:
            MINIBATCH_SIZE = batch_size
            # batch = random.sample(self.buffer, batch_size)

        sample_probs = self.get_probabilities(priority_scale)
        # print(sample_probs)
        sample_indices = random.choices(range(len(self.buffer)), k = MINIBATCH_SIZE, weights= sample_probs.tolist())
        batch = [self.buffer[i] for i in sample_indices]
        importance = self.get_importance(sample_probs[sample_indices])

        if only_state:
            res = torch.stack(tuple(item[3] for item in batch), dim=0)            
            return res.to(device)
        else:
            item_count = 5
            res = []
            for i in range(5):
                k = torch.stack(tuple(item[i] for item in batch), dim=0)
                res.append(k.to(device))
            return res[0], res[1], res[2], res[3], res[4], importance, sample_indices
    
    def set_priorities(self, indices, errors, offset=0.1):
        errors = errors.tolist()
        for i,e in zip(indices, errors):
            self.priorities[i] = abs(e[0]) + offset
