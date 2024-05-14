import os
import sys
import time
import pickle
import random
import datetime
from dataclasses import dataclass, field
from typing import Callable

from env import MahjongEnv
from model import GPTModel
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
MEMGET_NUM_PER_UPDATE = 20
REPLAY_BUFFER_SIZE = 2000
EPOCHES_PER_UPDATE = 1
ACTION_EPSILION = 0.2
MAX_UPDATE_KL = 1
CLIP_EPSILION = 0.4
WEIGHT_POLICY = 1
WEIGHT_VALUE = 1.5
N_BATCH_SAMPLE = 14
BATCH_SIZE = 60
EPISODES = 50
BTEA = 2
LR_VALUE = 8e-6
LR_POLICY = 1e-5
action_epsilion_scheduler = Linear_scheduler(50,ACTION_EPSILION,0.0025)

# 环境超参数
ACTION_TEMPERATURE = 0.1
STABLE_SEED_STEPS = 25# 保持种子在一定时间步内的稳定能增加拟合的可能?也更贴近少初始状态的RL。

# 目标超参数
TARGET = "N_step_TD"
LAMBD = 0.10
GAMMA = 0.99
ALPHA = 0.65
NTD = 4# TD自举步数
alpha_scheduler = Linear_scheduler(50,ALPHA,0.10)

# 模型超参数
SEPARATION_LAYER_POLICY = 5
SEPARATION_LAYER_VALUE = 4
DIM_FEEDFORWARD = 512
ENABLE_COMPILE = False
PAD_TOKEN_ID = 219
MAX_SEQ_LEN = 512
ACTION_SIZE = 185
VOCAB_SIZE = 1 + 184 + 34 + 1# [SEP]，action，手牌，[PAD]
MAX_NORM = 0.5
D_MODEL = 768# 内部feature维度
DROPOUT = 0.1
NHEAD = 8

# 计算参数&保存参数&显示参数
VERBOSE_POSITIVE_DONE_REWARD = True
REPLAY_BUFFER_FILE = "replay.pkl"
NDISPLAY = 10
LOG_DIR = f"runs/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
DEVICE = "cuda"
DTYPE = torch.float
PATH = "./mahjong"
PATH_MAX = "./max"

EVAL = False
if EVAL:
    PATH = PATH_MAX
    ACTION_EPSILION = 0
    MEMGET_NUM_PER_UPDATE = 1000
    NDISPLAY = int(MEMGET_NUM_PER_UPDATE/2)

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
    def __init__(self, lambd=LAMBD, gamma=GAMMA, eps_clip=CLIP_EPSILION, K_epochs=EPOCHES_PER_UPDATE, weight_policy=WEIGHT_POLICY, weight_value=WEIGHT_VALUE, alpha=alpha_scheduler, beta=BTEA) -> None:
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.weight_policy = weight_policy
        self.weight_value = weight_value
        self.alpha = alpha
        self.beta = beta
        self.lambd = lambd
        self.seed = 0
        self.seed_count = 0
        
        config = GPTModel.get_default_config()
        config.n_layer = SEPARATION_LAYER_POLICY
        config.n_head = NHEAD
        config.n_embd = D_MODEL
        config.vocab_size = VOCAB_SIZE
        config.out_size = 185
        config.block_size = 512
        self.policy_model = GPTModel(config)
        if os.path.exists(f"{PATH_MAX}_policy_max.pt"):
            self.policy_model.load_state_dict(torch.load(f"{PATH_MAX}_policy_max.pt"))
            print("Loaded")
        self.optimizer_policy = optim.AdamW(self.policy_model.parameters(), lr=LR_POLICY)
        self.policy_model_old = GPTModel(config)
        self.policy_model_old.load_state_dict(self.policy_model.state_dict())

        config.n_layer = SEPARATION_LAYER_VALUE
        config.out_size = 1
        self.value_model = GPTModel(config)
        if os.path.exists(f"{PATH_MAX}_value_max.pt"):
            self.value_model.load_state_dict(torch.load(f"{PATH_MAX}_value_max.pt"))
            print("Loaded")
        self.optimizer_value = optim.AdamW(self.value_model.parameters(), lr=LR_VALUE)
        self.value_model_old = GPTModel(config)
        self.value_model_old.load_state_dict(self.value_model.state_dict())  
        
        self.MseLoss = nn.MSELoss()
        
        if os.path.exists(REPLAY_BUFFER_FILE):
            with open(REPLAY_BUFFER_FILE,"rb") as file:
                self.replay_buffer = pickle.load(file)
        else:
            self.replay_buffer = ReplayBuffer()
        self.env = MahjongEnv()
        
        self.policy_model.to(DEVICE,DTYPE)
        self.policy_model_old.to(DEVICE,DTYPE)
        self.value_model.to(DEVICE,DTYPE)
        self.value_model_old.to(DEVICE,DTYPE)
        if sys.platform.startswith("linux") and ENABLE_COMPILE:
            self.policy_model = torch.compile(self.policy_model)
            self.policy_model_old = torch.compile(self.policy_model_old)
            self.value_model = torch.compile(self.value_model)
            self.value_model_old = torch.compile(self.value_model_old)
    
    def _get_GAEs(self, rewards, values) -> tensor:
        ending = torch.ones(len(rewards),device=DEVICE,dtype=DTYPE)
        ending[:-1] = 0
        # 计算delta：即时奖励加上折扣后的下一状态值函数，减去当前状态值函数
        values = torch.cat((values, torch.zeros(1,device=DEVICE,dtype=DTYPE)), dim=0)# 终端后值函数设为0
        deltas = rewards + self.gamma * values[1:] - values[:-1] * ending
        
        # 初始化GAE列表和累计优势
        gae = torch.zeros(1,device=DEVICE,dtype=DTYPE)
        advs = None
        
        # 逆向遍历deltas，计算GAE
        deltas.flip(0)
        for index, delta in enumerate(deltas[:-1]):
            gae = delta + self.gamma * self.lambd * gae * ending[index]
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
            nTD = torch.sum(rewards[index:index+NTD-1]*gammas[:-1])+values[index+NTD]*gammas[-1]
            nTD = nTD.unsqueeze(0)

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
        kls = []
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
                batch_logits = self.policy_model_old(old_states)[:,-1,:]
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
            batch_logits = self.policy_model(old_states)[:,-1,:]
            state_values = self.value_model(old_states)[:,-1,:]
            state_values = state_values.squeeze(1)
            
            batch_log_policy = None
            for index, logits in enumerate(batch_logits):
                if is_terminals[index]:
                    break
                logits = logits.squeeze(0)
                policy = softmax(logits,0).log()
                if batch_log_policy is None:
                    batch_log_policy = policy[actions[index]].unsqueeze(0)
                else:
                    batch_log_policy = torch.cat((batch_log_policy,policy[actions[index]].unsqueeze(0)))
            
            advantages = self._get_GAEs(tensor(rewards,device=DEVICE,dtype=DTYPE), state_values).detach()
            # 策略比例
            ratios = torch.exp(batch_log_policy-logprobs)
            kl = nn.functional.kl_div(logprobs,batch_log_policy,reduction='sum',log_target=True)
            kls.append(kl.item())
            if kl > MAX_UPDATE_KL:
                print(f"KL break\nKl:{kl}")
                break

            # PPO策略损失
            # [:-1]为不计入最后一步的策略损失
            # 策略裁剪(不使用是因为这似乎会导致梯度很容易无法传播)
            # cliped = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip)
            loss_policy = -ratios*advantages - self.beta*kl
            
            # 价值损失
            loss_value = self.MseLoss(state_values,
                                      (state_values+(self._get_nTDs(tensor(rewards,device=DEVICE,dtype=DTYPE),state_values)-state_values)*self.alpha.step()).detach()
                                      )
            loss_value = self.weight_value * loss_value
            loss_policy = self.weight_policy * loss_policy
            loss = loss_policy + loss_value
            loss_policy = loss_policy.mean()
            loss = loss.mean()
            loss = loss/int(BATCH_SIZE)#梯度累计正则化
            loss.backward()
            if not call_back is None:
                call_back(loss.mean().item(),loss_value.mean().item(),loss_policy.mean().item())
            
            # 优化&记录&梯度累计
            if (count+1)%int(BATCH_SIZE) == 0:
                for name, param in self.value_model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            print(f'NaN gradient in {name}')
                for name, param in self.policy_model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            print(f'NaN gradient in {name}')
                self.optimizer_policy.step()
                self.optimizer_policy.zero_grad()
                self.optimizer_value.step()
                self.optimizer_value.zero_grad()
                print(f"完成第{episode+1}轮{count+1}次权重更新")
            if (count+1)%int(BATCH_SIZE/NDISPLAY) == 0:
                print(f"Loss:{loss}")
                print(f"Value loss:{loss_value}")
                print(f"Policy loss:{loss_policy}")
                print(f"AVR Loss:{sum(display.losses)/len(display.losses)}\nAVR Value Loss:{sum(display.value_losses)/len(display.value_losses)}\nAVR Policy Loss:{sum(display.policy_losses)/len(display.policy_losses)}")
                print(f"AVR Policy kl:{sum(kls)/len(kls)}")
        
        self.policy_model_old.load_state_dict(self.policy_model.state_dict())
        self.value_model_old.load_state_dict(self.value_model.state_dict())
    
    def get_memory(self,call_back:Callable|None=None):
        done = False
        round = 0
        # 鉴于有四个智能体我们一次收集四分经验
        memories = [Trail() for _ in range(4)]
        random.seed(self.seed)
        self.seed_count += 1
        state, reward, done, info = self.env.reset()
        if self.seed_count >= STABLE_SEED_STEPS:
            self.seed_count = 0
            self.seed += 1
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
                logits = self.policy_model_old(input_ids)
            # 压缩batch
            logits = logits[:, -1, :]
            logits = logits.masked_fill(~tensor([True if item else False for item in action_mask],device=DEVICE),float('-inf'))# 去除无法使用的动作
            logits = logits[0]
            policy = softmax(logits/ACTION_TEMPERATURE,0)# 转概率密度
            
            # Action(ε贪婪)，而且我们显然不希望采样到同样的轨迹训练
            random.seed(time.time())
            if random.random() > action_epsilion_scheduler.get():
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
            if memory:
                self.replay_buffer.add(memory)

class Display:
    def __init__(self) -> None:
        self.rewards = []
        self.trail_rewards = []
        self.losses = []
        self.value_losses = []
        self.policy_losses = []
        
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
    
    def loss_update(self, loss:float, value_loss:float, policy_loss:float) -> None:
        self.losses.append(loss)
        self.value_losses.append(value_loss)
        self.policy_losses.append(policy_loss)
    
    def reset(self) -> None:
        self.writer.add_scalar('Loss/loss', sum(self.losses)/len(self.losses), self.step)
        self.writer.add_scalar('Loss/value loss', sum(self.value_losses)/len(self.value_losses), self.step)
        self.writer.add_scalar('Loss/policy loss', sum(self.policy_losses)/len(self.policy_losses), self.step)
        self.writer.add_scalar('AVR Reward', self.avr_reward, self.step)
        self.writer.add_scalar('AVR Trail Reward', self.avr_reward_per_trail, self.step)
        self.step += 1
        
        self.rewards = []
        self.trail_rewards = []
        self.losses = []
        self.value_losses = []
        self.policy_losses = []
        
        self.avr_reward = 0
        self.max_reward = 0
        self.avr_reward_per_trail = 0
        self.max_reward_per_trail = 0
             
if __name__ == "__main__":
    from rich import print
    
    agent = Agent()
    
    display = Display()
    current_max = -100
    for episode in range(EPISODES):
        for index in range(MEMGET_NUM_PER_UPDATE):
            agent.get_memory(display.reward_update)
            if (index+1) % int(MEMGET_NUM_PER_UPDATE/NDISPLAY) == 0:
                print(f"完成第{episode+1}轮{index+1}次轨迹收集\n平均奖励:{display.avr_reward:.4f}\n最大奖励:{display.max_reward:.4f}\n轨迹平均:{display.avr_reward_per_trail:.4f}\n轨迹最大:{display.max_reward_per_trail:.4f}")
        agent.update(agent.replay_buffer.sample(BATCH_SIZE*N_BATCH_SAMPLE),display.loss_update)
        print("Saving")
        if not EVAL:
            torch.save(agent.policy_model_old.state_dict(), f"{PATH}_policy.pt")
            torch.save(agent.value_model_old.state_dict(), f"{PATH}_value.pt")
            if display.avr_reward >= current_max:
                torch.save(agent.policy_model_old.state_dict(), f"{PATH_MAX}_policy_max.pt")
                torch.save(agent.value_model_old.state_dict(), f"{PATH_MAX}_value_max.pt")
                current_max = display.avr_reward
            with open(REPLAY_BUFFER_FILE,"wb") as file:
                pickle.dump(agent.replay_buffer, file)
        else:
            sys.exit()
        print("Saved")
        display.reset()
        action_epsilion_scheduler.step()