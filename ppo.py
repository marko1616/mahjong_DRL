import random
from dataclasses import dataclass, field
from typing import Callable

from env import MahjongEnv
from model import GPTModelWithValue

from collections import deque

import torch
from torch.nn.functional import softmax
from torch import tensor
from torch import optim
from torch import nn

# 训练超参数
MEMGET_NUM_PER_UPDATE = 500
REPLAY_BUFFER_SIZE = 10000
EPOCHES_PER_UPDATE = 2
ACTION_EPSILION = 0.1
CLIP_EPSILION = 0.2
WEIGHT_POLICY = 1
WEIGHT_VALUE = 1.2
BATCH_SIZE = 16
EPISODES = 100000
GAMMA = 1
LAMBD = 0.95
LR = 5e-5

# 模型超参数
NUM_DECODER_LAYERS = 2
DIM_FEEDFORWARD = 2048
DIM_VALUE_MLP = 4096
PAD_TOKEN_ID = 219
MAX_SEQ_LEN = 512
VOCAB_SIZE = 1 + 184 + 34 + 1# [SEP]，action，手牌，[PAD]
D_MODEL = 768# 内部feature维度
DROPOUT = 0.1
NHEAD = 12

# 计算参数&保存参数&显示参数
NDISPLAY = 5
DEVICE = "cuda"
DTYPE = torch.float
PATH = "./mahjong.pt"

@dataclass
class Trail:
    states: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    is_terminals: list = field(default_factory=list)
    info: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    action_logprobs:list = field(default_factory=list)

class ReplayBuffer:
    def __init__(self) -> None:
        self.buffer_size = REPLAY_BUFFER_SIZE
        self.buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    def add(self, trail: Trail) -> None:
        self.buffer.append(trail)

    def sample(self, batch_size) -> tuple:
        sample_size = min(len(self.buffer), batch_size)
        samples = random.sample(self.buffer, sample_size)
        return samples
    
class Agent:
    def __init__(self, lr=LR, gamma=GAMMA, lambd=LAMBD, eps_clip=CLIP_EPSILION, K_epochs=EPOCHES_PER_UPDATE, weight_policy=WEIGHT_POLICY, weight_value=WEIGHT_VALUE) -> None:
        self.gamma = gamma
        self.lambd = lambd
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.weight_policy = weight_policy
        self.weight_value = weight_value
        
        self.model = GPTModelWithValue(VOCAB_SIZE,D_MODEL,NHEAD,NUM_DECODER_LAYERS,DIM_FEEDFORWARD,DIM_VALUE_MLP,DROPOUT)
        self.model.load_state_dict(torch.load(PATH))
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model_old = GPTModelWithValue(VOCAB_SIZE,D_MODEL,NHEAD,NUM_DECODER_LAYERS,DIM_FEEDFORWARD,DIM_VALUE_MLP,DROPOUT)
        self.model_old.load_state_dict(self.model.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
        self.replay_buffer = ReplayBuffer()
        self.env = MahjongEnv()
        
        self.model.to(DEVICE)
        self.model_old.to(DEVICE)
        
        self.loss = 0
    
    def _get_GAEs(self, rewards, values) -> tensor:
        # 计算delta：即时奖励加上折扣后的下一状态值函数，减去当前状态值函数
        values = torch.cat([values, torch.tensor([0.0],dtype=DTYPE,device=DEVICE)], dim=0)# 终端后值函数设为0
        deltas = rewards + self.gamma * values[1:] - values[:-1]
        
        # 初始化GAE列表和累计优势
        gae = 0
        advs = []
        
        # 逆向遍历deltas，计算GAE
        for delta in reversed(deltas):
            gae = delta + self.gamma * self.lambd * gae
            advs.append(gae.tolist())
        advs.reverse()
        
        return torch.tensor(advs,device=DEVICE,dtype=DTYPE)

    def _get_TDλ(self, rewards, values) -> tensor:
        T = len(rewards)
        td_lambda_values = torch.zeros_like(rewards)
        future_rewards = 0
        for t in reversed(range(T)):
            td_error = rewards[t] + self.gamma * values[t + 1] - values[t] if t < T - 1 else rewards[t] - values[t]
            future_rewards = td_error + self.gamma * self.lambd * future_rewards
            td_lambda_values[t] = future_rewards
        return td_lambda_values
    
    def _random_one_index(self, multihot) -> int:
        # 找出所有值为1的索引
        zero_indices = [i for i, value in enumerate(multihot) if value == 1]
        # 随机选择一个索引
        if zero_indices:
            return random.choice(zero_indices)
        else:
            raise BaseException("Actions mask error")

    def _pad_list_to_length(self, lst, target_length=MAX_SEQ_LEN, pad_token_id=PAD_TOKEN_ID) -> list:
        return lst + [pad_token_id] * (target_length - len(lst))

    def update(self, memory: Trail) -> None:
        rewards = memory.rewards
        actions = memory.actions
        info = memory.info
        logprobs = tensor(memory.action_logprobs,device=DEVICE,dtype=DTYPE)
        
        # 获取需要的张量
        old_states = None
        for state in memory.states:
            if old_states is None:
                old_states = tensor(state,device=DEVICE).unsqueeze(0)
            else:
                old_states = torch.cat((old_states,tensor(state,device=DEVICE).unsqueeze(0)),dim=0)

        # 优化
        for _ in range(self.K_epochs):
            # 获取目前模型的评估
            batch_logits, state_values = self.model(old_states)
            state_values = state_values.squeeze(1)
            batch_log_policy = []
            for index, logits in enumerate(batch_logits):
                logits = logits.squeeze(0)
                logits = logits[:-35]# 去除牌型token和[PAD]
                logits = logits.masked_fill(~tensor([True if item else False for item in info[index]["action_mask"]],device=DEVICE),float('-inf'))# 去除无法使用的动作
                policy = softmax(logits,0).log()# 转对数概率密度
                batch_log_policy.append(policy[actions[index]].item())
            
            batch_log_policy = tensor(batch_log_policy,device=DEVICE,dtype=DTYPE)
            advantages = self._get_GAEs(tensor(rewards,device=DEVICE,dtype=DTYPE), state_values.detach())
            # 策略比例
            ratios = torch.exp(logprobs - batch_log_policy)
            #print(rewards)

            # PPO策略损失
            surr1 = ratios*advantages
            # 策略裁剪
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss_policy = -torch.min(surr1, surr2)
            
            # 价值损失
            loss_value = self.MseLoss(state_values,self._get_TDλ(tensor(rewards,device=DEVICE,dtype=DTYPE),state_values.detach()))
            
            loss = self.weight_policy * loss_policy + self.weight_value * loss_value
            
            # 优化
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        self.loss = loss.mean().item()
        
        self.model_old.load_state_dict(self.model.state_dict())
    
    def get_memory(self,call_back:Callable|None=None):
        done = False
        round = 0
        # 鉴于有四个智能体我们一次收集四分经验
        memories = [Trail()]*4
        state, reward, done, info = self.env.reset()
        while not done:
            action_mask = info["action_mask"]
            player_index = state["seat"]
            history = state["tokens"]
            hand = state["hand"]
            
            # 构建输入
            input_ids = []
            # 转换成token编码索引从185-219
            hand_tokens = []
            for index, num in enumerate(hand):
                hand_tokens += [185+index]*num
            input_ids += hand_tokens + [0] + history
            input_ids = tensor(input_ids,device=DEVICE)
            state = input_ids.tolist()
            input_ids = input_ids.unsqueeze(0)# 添加batch
            
            # 计算策略
            logits, value = self.model_old(input_ids)
            # 压缩batch
            logits = logits.squeeze(0)
            logits = logits[:-35]# 去除牌型token和[PAD]
            logits = logits.masked_fill(~tensor([True if item else False for item in action_mask],device=DEVICE),float('-inf'))# 去除无法使用的动作
            policy = softmax(logits,0)# 转概率密度
            value = value.squeeze(1)
            
            # Action(ε贪婪)
            if random.random() > ACTION_EPSILION:
                action = torch.multinomial(policy,1).item()
            else:
                action = self._random_one_index(action_mask)
            
            # 添加轨迹
            memories[player_index].states.append(self._pad_list_to_length(state))
            memories[player_index].rewards.append(reward)
            memories[player_index].is_terminals.append(done)
            memories[player_index].info.append(info)
            memories[player_index].actions.append(action)
            memories[player_index].action_logprobs.append(policy[action].item())
                
            state, reward, done, info = self.env.step(action)
            round += 1
            
            if not call_back is None:
                call_back(reward)
        
        for memory in memories:
            self.replay_buffer.add(memory)

class Display:
    def __init__(self) -> None:
        self.rewards = []
        
        self.avr_reward = 0
        self.max_reward = 0
    def update(self, reward:int) -> None:
        self.rewards.append(reward)
        
        self.avr_reward = sum(self.rewards)/len(self.rewards)
        self.max_reward = max(self.rewards)
    
    def reset(self) -> None:
        self.rewards = []
        
        self.avr_reward = 0
        self.max_reward = 0
             
if __name__ == "__main__":
    from rich import print
    
    agent = Agent()
    
    display = Display()
    for episode in range(EPISODES):
        for index in range(MEMGET_NUM_PER_UPDATE):
            agent.get_memory(display.update)
            if (index+1) % int(MEMGET_NUM_PER_UPDATE/NDISPLAY) == 0:
                print(f"完成第{episode+1}轮{index+1}次轨迹收集。平均奖励:{display.avr_reward:.4f}。最大奖励:{display.max_reward:.4f}")
        for index, memory in enumerate(agent.replay_buffer.sample(BATCH_SIZE)):
            agent.update(memory)
            if (index+1) % int(BATCH_SIZE/NDISPLAY) == 0:
                print(f"完成第{episode+1}轮{index+1}次权重更新\nLoss:{agent.loss}")
            torch.save(agent.model.state_dict(), PATH)
        display.reset()