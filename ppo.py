import os
import sys
import pickle
import random
import datetime
from dataclasses import dataclass, field
from typing import Callable

from env import MahjongEnv
from model import GPTModelWithValue
from schedulers import Linear_scheduler

from collections import deque

import torch
from torch.nn.functional import softmax
from torch.utils.tensorboard import SummaryWriter
from torch import tensor
from torch import optim
from torch import nn

now = datetime.datetime.now()

# 训练超参数
MEMGET_NUM_PER_UPDATE = 1200
REPLAY_BUFFER_SIZE = 6000
EPOCHES_PER_UPDATE = 2
ACTION_EPSILION = 0.025
CLIP_EPSILION = 0.2
WEIGHT_POLICY = 1
WEIGHT_VALUE = 1
N_BATCH_SAMPLE = 2
BATCH_SIZE = 1000
EPISODES = 1000
LR = 2.5e-9

# 目标超参数
TARGET = "N_step_TD"
LAMBD = 0.65
GAMMA = 0.99
ALPHA = 0.25
ALPHA_SCHEDULER = Linear_scheduler(100,ALPHA,0.01)
NTD = 2# TD自举步数

# 模型超参数
NUM_DECODER_LAYERS = 1
SEPARATION_LAYER= 2
DIM_FEEDFORWARD = 1024
ENABLE_COMPILE = False
PAD_TOKEN_ID = 219
MAX_SEQ_LEN = 512
ACTION_SIZE = 185
VOCAB_SIZE = 1 + 184 + 34 + 1# [SEP]，action，手牌，[PAD]
MAX_NORM = 0.5
D_MODEL = 1024# 内部feature维度
DROPOUT = 0.1
NHEAD = 8

# 计算参数&保存参数&显示参数
VERBOSE_POSITIVE_DONE_REWARD = True
REPLAY_BUFFER_FILE = "replay.pkl"
NDISPLAY = 10
LOG_DIR = f"runs/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
DEVICE = "cuda"
DTYPE = torch.float32
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
    def __init__(self, lr=LR, lambd=LAMBD, gamma=GAMMA, eps_clip=CLIP_EPSILION, K_epochs=EPOCHES_PER_UPDATE, weight_policy=WEIGHT_POLICY, weight_value=WEIGHT_VALUE, alpha=ALPHA_SCHEDULER) -> None:
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.weight_policy = weight_policy
        self.weight_value = weight_value
        self.alpha = alpha
        self.lambd = lambd
        
        self.model = GPTModelWithValue(VOCAB_SIZE,ACTION_SIZE,D_MODEL,NHEAD,NUM_DECODER_LAYERS,DIM_FEEDFORWARD,DROPOUT,SEPARATION_LAYER)
        if os.path.exists(PATH):
            self.model.load_state_dict(torch.load(PATH))
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.model_old = GPTModelWithValue(VOCAB_SIZE,ACTION_SIZE,D_MODEL,NHEAD,NUM_DECODER_LAYERS,DIM_FEEDFORWARD,DROPOUT,SEPARATION_LAYER)
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
        values = torch.cat((values, torch.zeros(1,device=DEVICE,dtype=DTYPE)), dim=0)# 终端后值函数设为0
        deltas = rewards + self.gamma * values[1:] - values[:-1]
        
        # 初始化GAE列表和累计优势
        gae = torch.zeros(1,device=DEVICE,dtype=DTYPE)
        advs = None
        
        # 逆向遍历deltas，计算GAE
        for delta in reversed(deltas):
            gae = delta + self.gamma * self.lambd * gae
            advs = gae if advs is None else torch.cat((advs,gae))
        advs.flip(0)
        
        return advs

    def _get_nTDs(self, rewards:tensor, values:tensor) -> tensor:
        trace_len = len(rewards)

        # 终端后值函数奖励设为0
        values = torch.cat((values, torch.zeros(NTD,device=DEVICE,dtype=DTYPE)), dim=0)
        rewards = torch.cat((rewards, torch.zeros(NTD,device=DEVICE,dtype=DTYPE)), dim=0)
        
        gammas = torch.pow(self.gamma, torch.arange(0, NTD, device=DEVICE,dtype=DTYPE))
        
        nTDs = None
        for index in range(trace_len):
            nTD = rewards[index:index+NTD-1]*gammas[:1]+values[index+NTD]*gammas[-1]

            nTDs = nTD if nTDs is None else torch.cat((nTDs,nTD))
        
        return nTDs
    
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

    def update(self, memories: Trail, call_back:Callable|None=None) -> None:
        for count, memory in enumerate(memories):
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
                policy = softmax(logits,0).log()
                if logprobs is None:
                    logprobs = policy[actions[index]].unsqueeze(0)
                else:
                    logprobs = torch.cat((logprobs,policy[actions[index]].unsqueeze(0)))

            # 优化
            # 获取目前模型的评估
            batch_logits, state_values = self.model(old_states,self.mask)
            state_values = state_values.squeeze(1)
            
            batch_log_policy = None
            for index, logits in enumerate(batch_logits):
                if is_terminals[index]:
                    break
                logits = logits.squeeze(0)
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
            loss_value = self.MseLoss(state_values,
                                      state_values+(self._get_nTDs(tensor(rewards,device=DEVICE,dtype=DTYPE),state_values)-state_values)*self.alpha.step()
                                      )
            loss = self.weight_policy * loss_policy + self.weight_value * loss_value
            loss.backward()
            
            # 优化&记录&梯度累计
            if (count+1)%BATCH_SIZE == 0:
                print(f"Loss:{loss}")
                self.optimizer.zero_grad()
                nn.utils.clip_grad_norm_(self.model.parameters(), MAX_NORM)
                if not call_back is None:
                    call_back(loss.mean().item(),loss_value.mean().item())
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            print(f'NaN gradient in {name}')
                self.optimizer.step()
                print(f"完成第{episode+1}轮{count+1}次权重更新\nAVR Loss:{sum(display.losses)/len(display.losses)}")
        
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
            # 更新向听奖励
            if len(memories[player_index].rewards)>=2:
                memories[player_index].rewards[-2] += info["reward_update"]
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
        self.value_losses = []
        
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
    
    def loss_update(self, loss:float, value_loss:float) -> None:
        self.losses.append(loss)
        self.value_losses.append(value_loss)
    
    def reset(self) -> None:
        self.writer.add_scalar('Loss/loss', sum(self.losses)/len(self.losses), self.step)
        self.writer.add_scalar('Loss/value loss', sum(self.value_losses)/len(self.value_losses), self.step)
        self.writer.add_scalar('AVR Reward', self.avr_reward, self.step)
        self.writer.add_scalar('AVR Trail Reward', self.avr_reward_per_trail, self.step)
        self.step += 1
        
        self.rewards = []
        self.trail_rewards = []
        self.losses = []
        self.value_losses = []
        
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
        agent.update(agent.replay_buffer.sample(BATCH_SIZE*N_BATCH_SAMPLE),display.loss_update)
        print("Saving")
        torch.save(agent.model.state_dict(), PATH)
        with open(REPLAY_BUFFER_FILE,"wb") as file:
            pickle.dump(agent.replay_buffer, file)
        print("Saved")
        display.reset()