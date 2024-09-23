import torch
from torch.utils.data import TensorDataset, DataLoader
from collections import deque

def rolling_average(values, window_size):
    window = deque(maxlen=window_size)
    averages = []
    
    for value in values:
        window.append(value)
        averages.append(sum(window) / len(window))
    
    return averages

def get_dataloader(df, batch_size, shuffle):
    inputs = torch.tensor(df['input'].tolist())
    labels = torch.tensor(df['label'].tolist())
    dataset = TensorDataset(inputs, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def loss_fn(logits, labels):
    if len(logits.shape)==3:
        logits = logits[:, -1]
    logits = logits.to(torch.float64)
    log_probs = logits.log_softmax(dim=-1)
    correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]
    return -correct_log_probs.mean()