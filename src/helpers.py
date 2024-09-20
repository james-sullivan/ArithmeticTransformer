import torch
from torch.utils.data import TensorDataset, DataLoader

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