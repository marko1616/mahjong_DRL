import torch
import torch.nn as nn


class GPTModelWithValue(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_decoder_layers: int, dim_feedforward: int = 2048, dim_value_mlp: int = 1024, dropout: float = 0.1):
        super(GPTModelWithValue, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 可训练的位置编码(反正牌局不会太长)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 512, d_model))
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)for _ in range(num_decoder_layers)
        ])
        self.value_head_mlp_1 = nn.Linear(d_model, dim_value_mlp)
        self.elu = nn.ELU()
        self.value_head = nn.Linear(dim_value_mlp, 1)  # 价值头，预测最后一个token的价值
        # 语言模型头，预测下一个token（action）
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, mask=None):
        x = self.embedding(input_ids) + self.positional_encoding[:, :input_ids.size(1), :]
        x = x.permute(1, 0, 2)  # 转换为(seq_length, batch_size, d_model)的形式

        for layer in self.decoder_layers:
            x = layer(x, x, tgt_mask=mask)

        x = x.permute(1, 0, 2)  # 转换回(batch_size, seq_length, d_model)的形式

        logits = self.lm_head(x[:, -1, :])  # 语言模型输出，用于action预测
        value_output = self.value_head_mlp_1(x[:, -1, :])  # 取每个序列最后一个token的输出来预测价值
        value_output = self.elu(value_output)
        value_output = self.value_head(value_output)

        return logits, value_output


if __name__ == "__main__":
    from rich import print

    vocab_size = 1 + 184 + 34 + 1  # [SEP]，action，手牌，<PAD>
    d_model = 768
    nhead = 12
    num_decoder_layers = 5

    model = GPTModelWithValue(vocab_size, d_model, nhead, num_decoder_layers)

    input_ids = torch.randint(0, vocab_size, (10, 100))

    logits, value_output = model(input_ids)
    print(f"input_ids Shape:{input_ids.shape}\nlogits Shape:{logits.shape}\nvalue_output Shape:{value_output.shape}")  # 输出形状
