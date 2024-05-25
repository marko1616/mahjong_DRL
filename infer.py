import torch
from config import *
from model import GPTModel
from rich.console import Console

from torch import tensor
from torch.nn.functional import softmax

SIGN_UPDATE = "MODEL_UPDATE"
SIGN_UPDATED = "MODEL_UPDATED"

def infer_engine(queue_in, queue_out, ids_in, config, device):
    try:
        model = GPTModel(config)
        model.to(device)
        model.eval()
        while True:
            if queue_in.qsize() > 0:
                command = queue_in.get()
                if command == SIGN_UPDATE:
                    model.load_state_dict(torch.load(f"{PATH}_policy.pt"))
                    model.to(device)
                    model.eval()
                    queue_out.put(SIGN_UPDATED)
            try:
                args = ids_in.get(timeout=1)
            except BaseException as e:
                continue
            out_queue = args[0]
            input_ids = args[1]
            action_mask = args[2]

            input_ids = tensor(input_ids,device=device)
            input_ids = input_ids.unsqueeze(0)# 添加batch
            # 计算策略
            with torch.no_grad():
                logits = model(input_ids)
            # 压缩batch
            logits = logits[:, -1, :]
            logits = logits.masked_fill(~tensor([True if item else False for item in action_mask],device=device),float('-inf'))# 去除无法使用的动作
            logits = logits[0]
            policy = softmax(logits/ACTION_TEMPERATURE,0)# 转概率密度
            action = torch.multinomial(policy,1).item()
            out_queue.put(action)
    except BaseException:
        Console.print_exception(show_locals=True)