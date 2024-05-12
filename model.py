import torch
import torch.nn as nn


class GPTModelWithValue(nn.Module):
    def __init__(self, vocab_size: int, action_size:int, d_model: int, nhead: int, num_decoder_layers: int, dim_feedforward: int = 2048, dropout: float = 0.1, separation_layer: int = 2, activation="gelu"):
        super(GPTModelWithValue, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 可训练的位置编码(反正牌局不会太长)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 512, d_model))
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation=activation)for _ in range(num_decoder_layers)
        ])

        # 价值头
        self.value_head = nn.Linear(d_model, 1)  # 价值头，预测最后一个token的价值
        # 语言模型头，预测下一个token（action）
        self.action_head = nn.Linear(d_model, action_size)
        
        # 策略头和价值头分离的transformer层
        self.separation_layer = separation_layer
        self.value_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation=activation)for _ in range(separation_layer)
        ])
        self.policy_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation=activation)for _ in range(separation_layer)
        ])

    def forward(self, input_ids, mask=None, no_value = False):
        x = self.embedding(input_ids) + self.positional_encoding[:, :input_ids.size(1), :]
        x = x.permute(1, 0, 2)  # 转换为(seq_length, batch_size, d_model)的形式

        for layer in self.decoder_layers:
            x = layer(x, x, tgt_mask=mask)
        
        policy = x
        for layer in self.policy_layers:
            policy = layer(policy, policy, tgt_mask=mask)
        policy = policy.permute(1, 0, 2)  # 转换回(batch_size, seq_length, d_model)的形式
        logits = self.action_head(policy[:, -1, :])  # 语言模型输出，用于action预测
        
        if not no_value:
            value = x
            for layer in self.value_layers:
                value = layer(value, value, tgt_mask=mask)
            value = value.permute(1, 0, 2)  # 转换回(batch_size, seq_length, d_model)的形式
            value_output = self.value_head(value[:, -1, :])
        else:
            value_output = None

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
