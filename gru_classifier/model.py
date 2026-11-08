import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GRUClassifier(nn.Module):
    """ids (B,T) -> emb (B,T,E) -> BiGRU -> feature (B,D*H) [last] or (B,3*D*H) [meanmax] -> linear (B,2)"""

    def __init__(self, vocab_size, embed_dim=128, hidden_size=128, num_layers=1,
                 bidirectional=True, dropout=0.3, num_classes=2, pad_id=0, pooling="last"):
        super().__init__()
        self.num_dirs = 2 if bidirectional else 1
        self.pooling = pooling
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.gru = nn.GRU(embed_dim, hidden_size, num_layers=num_layers, batch_first=True,
                          bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        feat = self.num_dirs * hidden_size * (3 if pooling == "meanmax" else 1)
        self.fc = nn.Linear(feat, num_classes)

    def forward(self, ids, lengths):
        emb = self.dropout(self.embedding(ids))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, h_n = self.gru(packed)
        last = torch.cat(list(h_n[-self.num_dirs:]), dim=1)
        if self.pooling == "meanmax":
            out, _ = pad_packed_sequence(out, batch_first=True)          # (B,T,D*H)
            mask = (torch.arange(out.size(1))[None, :] < lengths[:, None]).to(out.device).unsqueeze(-1)
            mean = (out * mask).sum(1) / lengths.to(out.device).unsqueeze(1)
            mx = out.masked_fill(~mask, -1e4).max(1).values
            last = torch.cat([last, mean, mx], dim=1)
        return self.fc(self.dropout(last))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
