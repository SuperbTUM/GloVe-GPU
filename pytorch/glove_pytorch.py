import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from tqdm import tqdm
import time


class Model(nn.Module):
    def __init__(self, vocab_size=251, context_len=4, embedding_dim=16, hidden_dim=128):
        super(Model, self).__init__()
        self.vocab_size = vocab_size
        self.context_len = context_len
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_dim)
        self.hidden = nn.Linear(embedding_dim * context_len, hidden_dim, bias=True)
        self.hidden_activated = nn.Sigmoid()
        self.output = nn.Linear(hidden_dim, vocab_size * context_len, bias=True)
        self.output_activated = nn.Softmax(dim=-1)  # (B, NV)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, std=0.01)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight.data, std=0.01)

    def forward(self, x):
        emb = torch.zeros((x.shape[0], self.embedding_dim * self.context_len))
        for i in range(self.context_len):
            emb[:, self.embedding_dim * i: (i+1) * self.embedding_dim] = self.embedding(x)[:, i, :]
        hid = self.hidden(emb)
        hid_ac = self.hidden_activated(hid)
        out = self.output(hid_ac)
        shape_out = out.shape
        out = out.reshape(-1, shape_out[1] // self.vocab_size, self.vocab_size)
        out_ac = self.output_activated(out)
        return out_ac.reshape(shape_out)


class GloVeDataset(Dataset):
    def __init__(self, data):
        super(GloVeDataset, self).__init__()
        self.train_inputs = data["train_inputs"]

    def __len__(self):
        if isinstance(self.train_inputs, np.ndarray):
            return self.train_inputs.shape[0]
        else:
            return len(self.train_inputs)

    def __getitem__(self, item):
        context = self.train_inputs[item]
        return context


class CELoss(nn.Module):
    def __init__(self, reduction="sum"):
        super().__init__()
        self.reduction = reduction

    def forward(self, target_batch, output_activated):
        if self.reduction == "sum":
            return -(target_batch * torch.log(output_activated + 1e-6)).sum()
        elif self.reduction == "mean":
            return -(target_batch * torch.log(output_activated + 1e-6)).sum() / target_batch.shape[0]


def sample_input_mask(batch_size, context_length):
    """Samples a binary mask for the inputs of size batch_size x context_len
    For each row, at most one element will be 1.
    """
    mask_idx = torch.randint(context_length, size=(batch_size,))
    mask = torch.zeros((batch_size, context_length),
                       dtype=torch.int32)  # Convert to one hot B x N, B batch size, N context len
    mask[torch.arange(batch_size), mask_idx] = 1
    return mask


def indicator_matrix(input_masked, mask_zero=True):
    batch_size, context_length = input_masked.shape
    target_batch = torch.zeros((batch_size, context_length * vocab_size))
    targets_offset = (torch.arange(context_length) * vocab_size).unsqueeze(0).repeat_interleave(batch_size, dim=0)
    input_masked += targets_offset
    input_masked = input_masked.numpy()
    target_batch = target_batch.numpy()
    targets_offset = targets_offset.numpy()
    for c in range(context_length):
        target_batch[np.arange(batch_size), input_masked[:, c]] = 1.
        if mask_zero:
            target_batch[np.arange(batch_size), targets_offset[:, c]] = 0.
    return torch.Tensor(target_batch)


if __name__ == "__main__":
    start = time.time()
    epochs = 2
    batch_size = 100
    data = pickle.load(open("data.pk", "rb"))
    vocab_size = len(data["vocab"])
    data_inputs = data["train_inputs"]
    context_length = data["train_inputs"].shape[1]
    mask = sample_input_mask(batch_size, context_length)
    model = Model(vocab_size)  # .cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=.5)
    losses = []
    for epoch in range(epochs):
        loss_avg = 0.
        dataloader = DataLoader(GloVeDataset(data), batch_size=batch_size, shuffle=True,
                                num_workers=4, pin_memory=True)
        for sample in tqdm(dataloader):
            with torch.no_grad():
                input_masked = sample * (1 - mask)
                target_masked = sample * mask
                input_masked_expand = indicator_matrix(target_masked)
            # input_masked_expand = input_masked_expand.cuda()
            optimizer.zero_grad()
            output = model(input_masked)
            loss = CELoss(reduction="mean")(input_masked_expand, output)
            loss.backward()
            loss_avg += loss.detach().numpy()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            lr_scheduler.step()
        losses.append(loss_avg / (data_inputs.shape[0] // batch_size))
    end = time.time()
    print("PyTorch CPU implementation time is {:.3f} seconds.".format(end-start))
    print("loss ", losses)
