import os
import sys
import pickle
import random
import datetime
from dataclasses import dataclass, field
from typing import Callable

from env import MahjongEnv
from model import GPTModelWithValue

from collections import deque

import torch
from torch.nn.functional import softmax
from torch.utils.tensorboard import SummaryWriter
from torch import tensor
from torch import optim
from torch import nn

now = datetime.datetime.now()

# 训练超参数
MEMGET_NUM_PER_UPDATE = 400
REPLAY_BUFFER_SIZE = 2500
EPOCHES_PER_UPDATE = 10
ACTION_EPSILION = 0.1
CLIP_EPSILION = 0.2
WEIGHT_POLICY = 1
WEIGHT_VALUE = 1.2
BATCH_SIZE = 200
EPISODES = 100000
GAMMA = 1
ALPHA = 0.25
LAMBD = 0.45
LR = 5e-5

# 模型超参数
NUM_DECODER_LAYERS = 4
SEPARATION_LAYER= 2
DIM_FEEDFORWARD = 2048
ENABLE_COMPILE = False
DIM_VALUE_MLP = 4096
PAD_TOKEN_ID = 219
MAX_SEQ_LEN = 512
VOCAB_SIZE = 1 + 184 + 34 + 1# [SEP]，action，手牌，[PAD]
MAX_NORM = 0.5
D_MODEL = 1024# 内部feature维度
DROPOUT = 0.1
NHEAD = 8

# 计算参数&保存参数&显示参数
VERBOSE_POSITIVE_DONE_REWARD = True
REPLAY_BUFFER_FILE = "replay.pkl"
NDISPLAY = 4
LOG_DIR = f"runs/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
DEVICE = "cuda"
DTYPE = torch.float
PATH = "./mahjong.pt"

def generate_square_subsequent_mask(sz,device):
    mask = (torch.triu(torch.ones(sz, sz, device=device, dtype=DTYPE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

@dataclass
class Trail:
    states: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    is_terminals: list = field(default_factory=list)
    info: list = field(default_factory=list)
    actions: list = field(default_factory=list)

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
    def __init__(self, lr=LR, gamma=GAMMA, lambd=LAMBD, eps_clip=CLIP_EPSILION, K_epochs=EPOCHES_PER_UPDATE, weight_policy=WEIGHT_POLICY, weight_value=WEIGHT_VALUE, alpha=ALPHA) -> None:
        self.gamma = gamma
        self.lambd = lambd
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.weight_policy = weight_policy
        self.weight_value = weight_value
        self.alpha = alpha
        
        self.model = GPTModelWithValue(VOCAB_SIZE,D_MODEL,NHEAD,NUM_DECODER_LAYERS,DIM_FEEDFORWARD,DROPOUT,SEPARATION_LAYER)
        if os.path.exists(PATH):
            self.model.load_state_dict(torch.load(PATH))
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.model_old = GPTModelWithValue(VOCAB_SIZE,D_MODEL,NHEAD,NUM_DECODER_LAYERS,DIM_FEEDFORWARD,DROPOUT,SEPARATION_LAYER)
        self.model_old.load_state_dict(self.model.state_dict())
        self.mask = generate_square_subsequent_mask(MAX_SEQ_LEN,DEVICE)
        
        self.MseLoss = nn.MSELoss()
        
        if os.path.exists(REPLAY_BUFFER_FILE):
            with open(REPLAY_BUFFER_FILE,"rb") as file:
                self.replay_buffer = pickle.load(file)
        else:
            self.replay_buffer = ReplayBuffer()
        self.env = MahjongEnv()
        
        self.model.to(DEVICE,DTYPE)
        self.model_old.to(DEVICE,DTYPE)
        if sys.platform.startswith("linux") and ENABLE_COMPILE:
            self.model = torch.compile(self.model)
            self.model_old = torch.compile(self.model_old)
    
    def _get_GAEs(self, rewards, values) -> tensor:
        # 计算delta：即时奖励加上折扣后的下一状态值函数，减去当前状态值函数
        values = torch.cat([values, tensor([0.0],dtype=DTYPE,device=DEVICE)], dim=0)# 终端后值函数设为0
        deltas = rewards + self.gamma * values[1:] - values[:-1]
        
        # 初始化GAE列表和累计优势
        gae = 0
        advs = []
        
        # 逆向遍历deltas，计算GAE
        for delta in reversed(deltas):
            gae = delta + self.gamma * self.lambd * gae
            advs.append(gae.item())
        advs.reverse()
        
        return tensor(advs,device=DEVICE,dtype=DTYPE)

    def _get_TDλ(self, rewards:tensor, values:tensor) -> tensor:
        values = torch.cat([values, tensor([0.0],dtype=DTYPE,device=DEVICE)], dim=0)# 终端后值函数设为0
        
        n = rewards.size(0)
        deltas = rewards + self.gamma * values[1:] - values[:-1]
        e = torch.zeros_like(values)
        value_updates = torch.zeros_like(values)
        
        for t in range(n-1):
            e[t+1] = self.gamma * self.lambd * e[t]
            e[t] += 1.0
            value_updates += self.alpha * deltas[t] * e
        
        return (values + value_updates)[:-1]
    
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

    def update(self, memory: Trail, call_back:Callable|None=None) -> None:
        rewards = memory.rewards
        actions = memory.actions
        is_terminals = memory.is_terminals
        info = memory.info
        
        # 获取需要的张量
        old_states = None
        for state in memory.states:
            if old_states is None:
                old_states = tensor(state,device=DEVICE).unsqueeze(0)
            else:
                old_states = torch.cat((old_states,tensor(state,device=DEVICE).unsqueeze(0)),dim=0)
                
        # 获取当前在线模型的动作对数概率
        logprobs = None
        with torch.no_grad():
            batch_logits, _ = self.model_old(old_states,self.mask)
        for index, logits in enumerate(batch_logits):
            if is_terminals[index]:
                break
            logits = logits.squeeze(0)
            logits = logits[:-35]# 去除牌型token和[PAD]
            policy = softmax(logits,0).log()
            if logprobs is None:
                logprobs = policy[actions[index]].unsqueeze(0)
            else:
                logprobs = torch.cat((logprobs,policy[actions[index]].unsqueeze(0)))

        # 优化
        for _ in range(self.K_epochs):
            # 获取目前模型的评估
            batch_logits, state_values = self.model(old_states,self.mask)
            state_values = state_values.squeeze(1)
            
            batch_log_policy = None
            for index, logits in enumerate(batch_logits):
                if is_terminals[index]:
                    break
                logits = logits.squeeze(0)
                logits = logits[:-35]# 去除牌型token和[PAD]
                policy = softmax(logits,0).log()# 转对数概率密度
                if batch_log_policy is None:
                    batch_log_policy = policy[actions[index]].unsqueeze(0)
                else:
                    batch_log_policy = torch.cat((batch_log_policy,policy[actions[index]].unsqueeze(0)))
            
            advantages = self._get_GAEs(tensor(rewards,device=DEVICE,dtype=DTYPE), state_values).detach()
            # 策略比例
            ratios = torch.exp(batch_log_policy-logprobs)

            # PPO策略损失
            # [:-1]为不计入最后一步的策略损失
            # 策略裁剪
            cliped = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip)
            loss_policy = -torch.min(ratios*advantages[:-1], cliped*advantages[:-1]).mean()
            
            # 价值损失
            loss_value = self.MseLoss(state_values,self._get_TDλ(tensor(rewards,device=DEVICE,dtype=DTYPE),state_values).detach())
            loss = self.weight_policy * loss_policy + self.weight_value * loss_value
            
            # 优化&记录
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), MAX_NORM)
            if not call_back is None:
                call_back(loss.mean().item())
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        print(f'NaN gradient in {name}')
            self.optimizer.step()
        
        self.model_old.load_state_dict(self.model.state_dict())
    
    def get_memory(self,call_back:Callable|None=None):
        done = False
        round = 0
        # 鉴于有四个智能体我们一次收集四分经验
        memories = [Trail() for _ in range(4)]
        state, reward, done, info = self.env.reset()
        no_memory_index = []
        while True:
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
            input_ids = self._pad_list_to_length(input_ids)
            memories[player_index].states.append(input_ids)
            input_ids = tensor(input_ids,device=DEVICE)
            input_ids = input_ids.unsqueeze(0)# 添加batch
            
            # 添加轨迹
            memories[player_index].rewards.append(reward)
            memories[player_index].is_terminals.append(done)
            memories[player_index].info.append(info)
            if done:
                # 方便截断的填充
                memories[player_index].actions.append(0)
                if VERBOSE_POSITIVE_DONE_REWARD and reward > 5:
                    print(f"终局回报{reward}")
                # 为所有人添加终端状态
                for index, memory in enumerate(memories):
                    if memory.is_terminals:
                        memory.is_terminals[-1] = True
                    else:
                        # 天胡情况处理
                        no_memory_index.append(index)
                break
            
            # 计算策略
            with torch.no_grad():
                logits, _ = self.model_old(input_ids,self.mask,no_value=True)
            # 压缩batch
            logits = logits.squeeze(0)
            logits = logits[:-35]# 去除牌型token和[PAD]
            logits = logits.masked_fill(~tensor([True if item else False for item in action_mask],device=DEVICE),float('-inf'))# 去除无法使用的动作
            policy = softmax(logits,0)# 转概率密度
            
            # Action(ε贪婪)
            if random.random() > ACTION_EPSILION:
                action = torch.multinomial(policy,1).item()
            else:
                action = self._random_one_index(action_mask)
            
            memories[player_index].actions.append(action)
                
            state, reward, done, info = self.env.step(action)
            round += 1
            
        if not call_back is None:
            for index, memory in enumerate(memories):
                if not index in no_memory_index:
                    call_back(memory.rewards)
        
        for memory in memories:
            self.replay_buffer.add(memory)

class Display:
    def __init__(self) -> None:
        self.rewards = []
        self.trail_rewards = []
        self.losses = []
        
        self.avr_reward = 0
        self.max_reward = 0
        self.avr_reward_per_trail = 0
        self.max_reward_per_trail = 0
        
        self.step = 0
        self.writer = SummaryWriter(LOG_DIR)
    def reward_update(self, reward_trail: list[int]) -> None:
        for reward in reward_trail:
            self.rewards.append(reward)
        self.trail_rewards.append(sum(reward_trail))
        
        self.avr_reward = sum(self.rewards)/len(self.rewards)
        self.max_reward = max(self.rewards)
        self.avr_reward_per_trail = sum(self.trail_rewards)/len(self.trail_rewards)
        self.max_reward_per_trail = max(self.trail_rewards)
    
    def loss_update(self, loss:float) -> None:
        self.losses.append(loss)
    
    def reset(self) -> None:
        self.writer.add_scalar('Loss/train', sum(self.losses)/len(self.losses), self.step)
        self.writer.add_scalar('AVR Reward', self.avr_reward, self.step)
        self.writer.add_scalar('AVR Trail Reward', self.avr_reward_per_trail, self.step)
        self.step += 1
        
        self.rewards = []
        self.trail_rewards = []
        self.losses = []
        
        self.avr_reward = 0
        self.max_reward = 0
        self.avr_reward_per_trail = 0
        self.max_reward_per_trail = 0
             
if __name__ == "__main__":
    from rich import print
    
    agent = Agent()
    
    display = Display()
    for episode in range(EPISODES):
        for index in range(MEMGET_NUM_PER_UPDATE):
            agent.get_memory(display.reward_update)
            if (index+1) % int(MEMGET_NUM_PER_UPDATE/NDISPLAY) == 0:
                print(f"完成第{episode+1}轮{index+1}次轨迹收集\n平均奖励:{display.avr_reward:.4f}\n最大奖励:{display.max_reward:.4f}\n轨迹平均:{display.avr_reward_per_trail:.4f}\n轨迹最大:{display.max_reward_per_trail:.4f}")
        for index, memory in enumerate(agent.replay_buffer.sample(BATCH_SIZE)):
            agent.update(memory,display.loss_update)
            if (index+1) % int(BATCH_SIZE/NDISPLAY) == 0:
                print(f"完成第{episode+1}轮{index+1}次权重更新\nAVR Loss:{sum(display.losses)/len(display.losses)}")
        print("Saving")
        torch.save(agent.model.state_dict(), PATH)
        with open(REPLAY_BUFFER_FILE,"wb") as file:
            pickle.dump(agent.replay_buffer, file)
        print("Saved")
        display.reset()