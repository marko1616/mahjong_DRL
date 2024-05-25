import os
import sys
import time
import pickle
import random
import psutil
import asyncio
import datetime
import multiprocessing
from typing import Callable
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor

from env import env_process, action_to_str, RESET_SIGN, STOP_SIGN
from model import set_seeds, GPTModel
from infer import infer_engine, SIGN_UPDATE, SIGN_UPDATED
from config import *

from collections import deque

from rich import print
from rich.console import Console

import torch
from torch.nn.functional import softmax
from torch.utils.tensorboard import SummaryWriter
from torch import tensor
from torch import optim
from torch import nn

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
        set_seeds()# 回放采样还是随机一点吧
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
        self.seed = INIT_ENV_SEED
        self.seed_count = 0
        self.max_trail_len = 0
        
        config = GPTModel.get_default_config()
        config.n_layer = SEPARATION_LAYER_POLICY
        config.n_head = NHEAD
        config.n_embd = D_MODEL
        config.vocab_size = VOCAB_SIZE
        config.out_size = 185
        config.block_size = 512
        self.policy_model = GPTModel(config)
        if os.path.exists(f"{PATH}_policy.pt"):
            self.policy_model.load_state_dict(torch.load(f"{PATH_MAX}_policy.pt"))
            print("Policy model loaded")
        self.optimizer_policy = optim.AdamW(self.policy_model.parameters(), lr=LR_POLICY)
        self.policy_model_old = GPTModel(config)
        self.policy_model_old.load_state_dict(self.policy_model.state_dict())
        self.policy_model_old.eval()
        # 推理阶段允许多卡并行
        self.infer_manager = multiprocessing.Manager()
        self.infer_ids_in_queue = self.infer_manager.Queue()
        self.infer_in_queues = [self.infer_manager.Queue() for _ in range(len(INFER_DEVICES))]
        self.infer_out_queues = [self.infer_manager.Queue() for _ in range(len(INFER_DEVICES))]
        self.infer_executor = ProcessPoolExecutor()
        for index, device in enumerate(INFER_DEVICES):
            self.infer_executor.submit(infer_engine,queue_in=self.infer_in_queues[index],queue_out=self.infer_out_queues[index],ids_in=self.infer_ids_in_queue,config=config,device=device)

        config = GPTModel.get_default_config()
        config.n_layer = SEPARATION_LAYER_VALUE
        config.n_head = NHEAD
        config.n_embd = D_MODEL
        config.vocab_size = VOCAB_SIZE
        config.out_size = 1
        config.block_size = 512
        self.value_model = GPTModel(config)
        if os.path.exists(f"{PATH}_value.pt"):
            self.value_model.load_state_dict(torch.load(f"{PATH_MAX}_value.pt"))
            print("Value model loaded")
        self.optimizer_value = optim.AdamW(self.value_model.parameters(), lr=LR_VALUE)
        self.value_model_old = GPTModel(config  )
        self.value_model_old.load_state_dict(self.value_model.state_dict())  
        self.value_model_old.eval()
        
        self.MseLoss = nn.MSELoss()
        
        if os.path.exists(REPLAY_BUFFER_FILE):
            with open(REPLAY_BUFFER_FILE,"rb") as file:
                self.replay_buffer = pickle.load(file)
                print("Replay buffer loaded")
        else:
            self.replay_buffer = ReplayBuffer()
        
        self.policy_model.to(DEVICE,DTYPE)
        self.policy_model_old.to(DEVICE,DTYPE)
        self.value_model.to(DEVICE,DTYPE)
        self.value_model_old.to(DEVICE,DTYPE)
        if sys.platform.startswith("linux") and ENABLE_COMPILE:
            self.policy_model = torch.compile(self.policy_model)
            self.policy_model_old = torch.compile(self.policy_model_old)
            self.value_model = torch.compile(self.value_model)
            self.value_model_old = torch.compile(self.value_model_old)
        
        # 我也不想写多进程啊，奈何单核CPU过慢，显卡还不能吃满
        self.env_manager = multiprocessing.Manager()
        self.env_out_queues = [self.env_manager.Queue() for _ in range(ENV_NUM)]
        self.env_in_queues = [self.env_manager.Queue() for _ in range(ENV_NUM)]
        self.env_executor = ProcessPoolExecutor()
        for index in range(ENV_NUM):
            self.env_executor.submit(env_process,queue_in=self.env_in_queues[index],queue_out=self.env_out_queues[index])
        
        self.event_loop = asyncio.new_event_loop()
        # 只能同时有一个任务占用环境进程
        self.conditions = [asyncio.Condition() for _ in range(ENV_NUM)]
        self.env_lock = [0]*ENV_NUM
    
    def _get_GAEs(self, rewards, values) -> tensor:
        ending = torch.ones(len(rewards),device=DEVICE,dtype=DTYPE)
        ending[:-1] = 0
        values = torch.cat((values, torch.zeros(1,device=DEVICE,dtype=DTYPE)), dim=0)# 终端后值函数设为0
        # 计算delta：即时奖励加上折扣后的下一状态值函数，减去当前状态值函数
        deltas = rewards + self.gamma * values[1:] - values[:-1] * ending
        
        # 初始化GAE列表和累计优势
        gae = torch.zeros(1,device=DEVICE,dtype=DTYPE)
        advs = None
        
        # 逆向遍历deltas，计算GAE
        deltas.flip(0)
        for index, delta in enumerate(deltas):
            gae = delta + self.gamma * self.lambd * gae * ending[index]
            advs = gae if advs is None else torch.cat((advs,gae))
        advs.flip(0)
        
        return advs

    def _get_nTDs(self, rewards:tensor, values:tensor) -> tensor:
        ntd_now = int(n_td_scheduler.get())
        trace_len = len(rewards)

        # 终端后值函数奖励设为0
        values = torch.cat((values, torch.zeros(ntd_now,device=DEVICE,dtype=DTYPE)), dim=0)# 终端后值函数设为0
        rewards = torch.cat((rewards, torch.zeros(ntd_now,device=DEVICE,dtype=DTYPE)), dim=0)
        
        gammas = torch.pow(self.gamma, torch.arange(0, ntd_now, device=DEVICE,dtype=DTYPE))
        
        nTDs = None
        for index in range(trace_len):
            nTD = torch.sum(rewards[index:index+ntd_now-1]*gammas[:-1])+values[index+ntd_now]*gammas[-1]
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
            assert len(memory.states) - 1 == len(memory.rewards)
            for state in memory.states:
                if old_states is None:
                    old_states = tensor(state,device=DEVICE).unsqueeze(0)
                else:
                    old_states = torch.cat((old_states,tensor(state,device=DEVICE).unsqueeze(0)),dim=0)
            old_states = old_states[:-1]
                    
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
            state_values_old = self.value_model_old(old_states)[:,-1,:]
            state_values_old = state_values_old.squeeze(1)
            
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
            
            advantages = self._get_GAEs(tensor(rewards,device=DEVICE,dtype=DTYPE), state_values_old).detach()
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
            loss_policy = loss_policy.mean()
            
            # 价值损失
            state_values = self.value_model(old_states)[:,-1,:]
            state_values = state_values.squeeze(1)
            loss_value = self.MseLoss(state_values,
                                      (state_values+(self._get_nTDs(tensor(rewards,device=DEVICE,dtype=DTYPE),state_values)-state_values)*self.alpha.step()).detach()
                                      )
            loss_value = self.weight_value * loss_value / int(BATCH_SIZE)#梯度累计正则化
            loss_policy = self.weight_policy * loss_policy / int(BATCH_SIZE)#梯度累计正则化
            loss_value.backward()
            loss_policy.backward()
            loss = (loss_value + loss_policy).detach()
            loss = loss.mean()
            if call_back is not None:
                call_back(loss.item(),loss_value.item(),loss_policy.item())
            
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
        self.policy_model_old.eval()
        self.value_model_old.eval()
    
    def _get_pad_ids(self,hand,history) -> list:
        # 构建输入
        input_ids = []
        # 转换成token编码索引从185-219
        hand_tokens = []
        for index, num in enumerate(hand):
            hand_tokens += [185+index]*num
        input_ids += hand_tokens + [0] + history
        input_ids = self._pad_list_to_length(input_ids)
        return input_ids

    async def get_memory(self,process_id:int,call_back:Callable|None=None) -> None:
        # 环境锁
        env_id = process_id % ENV_NUM
        async with self.conditions[env_id]:
            await self.conditions[env_id].wait_for(lambda: self.env_lock[env_id] == int(process_id/ENV_NUM))

        done = False
        current_round = 0
        # 鉴于有四个智能体我们一次收集四分经验
        memories = [Trail([],[],[],[],[]) for _ in range(4)]
        if not EVAL:
            set_seeds(self.seed)
        else:
            set_seeds()
        self.seed_count += 1
        self.env_in_queues[env_id].put(RESET_SIGN)
        state, reward, done, info = await self.event_loop.run_in_executor(None,self.env_out_queues[env_id].get)
        if self.seed_count >= STABLE_SEED_STEPS:
            self.seed_count = 0
            self.seed += 1
        no_memory_index = []
        result_get_queue = self.infer_manager.Queue()
        while True:
            action_mask = info["action_mask"]
            player_index = state["seat"]
            history = state["tokens"]
            hand = state["hand"]
            
            input_ids = self._get_pad_ids(hand,history)
            memories[player_index].states.append(input_ids)
            
            # Action(ε贪婪)，而且我们显然不希望采样到同样的轨迹训练
            set_seeds()
            if random.random() > action_epsilion_scheduler.get():
                # 不需要计算如果没被选择使用策略的话
                self.infer_ids_in_queue.put((result_get_queue,input_ids,action_mask))
                action = await self.event_loop.run_in_executor(None,result_get_queue.get)
            else:
                action = self._random_one_index(action_mask)

            self.env_in_queues[env_id].put(action)
            state, reward, done, info = await self.event_loop.run_in_executor(None,self.env_out_queues[env_id].get)

            # 添加轨迹
            memories[player_index].is_terminals.append(False)
            memories[player_index].info.append(info)
            memories[player_index].actions.append(action)
            memories[player_index].rewards.append(reward)
            # 更新向听奖励
            if len(memories[player_index].rewards)>=2:
                memories[player_index].rewards[-2] += info["reward_update"]
            if done:
                # 方便截断的填充
                if VERBOSE_POSITIVE_DONE_REWARD and reward > 5:
                    print(f"终局回报{reward}")
                # 为所有人添加终端状态
                for index, memory in enumerate(memories):
                    memory.states.append([PAD_TOKEN_ID]*MAX_SEQ_LEN)

                    if memory.is_terminals:
                        memory.is_terminals.append(True)
                    else:
                        # 天胡等有人打不了牌情况处理
                        no_memory_index.append(index)
                break
            current_round += 1
            
        if call_back is not None:
            for index, memory in enumerate(memories):
                if index not in no_memory_index:
                    call_back(memory.rewards)
        
        for index, memory in enumerate(memories):
            if index not in no_memory_index:
                if len(memory.states) > self.max_trail_len:
                    print(f"Max trail len up to {len(memory.states)}")
                    self.max_trail_len = len(memory.states)
                self.replay_buffer.add(memory)

        # 释放环境给队列的下一个任务
        async with self.conditions[env_id]:
            self.env_lock[env_id] += 1
            self.conditions[env_id].notify_all()
    
    def get_memory_batch(self,call_back:Callable|None=None) -> None:
        tasks = []
        for index in range(int(MEMGET_NUM_PER_UPDATE/4)):
            tasks.append(self.event_loop.create_task(self.get_memory(index,call_back)))
        self.event_loop.run_until_complete(asyncio.gather(*tasks))

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
        self.reward_update_time = 0
        
        self.step = 0
        self.writer = SummaryWriter(LOG_DIR)
    def reward_update(self, reward_trail: list[int]) -> None:
        for reward in reward_trail:
            self.rewards.append(reward)
        self.trail_rewards.append(sum(reward_trail))
        self.reward_update_time += 1
        
        self.avr_reward = sum(self.rewards)/len(self.rewards)
        self.max_reward = max(self.rewards)
        self.avr_reward_per_trail = sum(self.trail_rewards)/len(self.trail_rewards)
        self.max_reward_per_trail = max(self.trail_rewards)

        if self.reward_update_time % int(MEMGET_NUM_PER_UPDATE/NDISPLAY) == 0:
            print(f"完成第{self.step+1}轮{self.reward_update_time}次轨迹收集\n平均奖励:{self.avr_reward:.4f}\n最大奖励:{self.max_reward:.4f}\n轨迹平均:{self.avr_reward_per_trail:.4f}\n轨迹最大:{self.max_reward_per_trail:.4f}")
    
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
        self.reward_update_time = 0

def kill_child_processes(parent_pid):
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        # 找不到自己可还行
        return
    children = parent.children(recursive=True)
    for process in children:
        process.kill()

if __name__ == "__main__":
    if sys.platform.startswith('linux'):
        # 无法使用fork(说实话创建一堆进程各个占用几个GB的内存还真挺抽象的)
        multiprocessing.set_start_method("spawn")
    now = datetime.datetime.now()
    time_start = time.time()
    assert MEMGET_NUM_PER_UPDATE % 4 == 0
    try:
        agent = Agent()
        display = Display()
        current_max = -1000
        for episode in range(EPISODES):
            for queue_in in agent.infer_in_queues:
                queue_in.put(SIGN_UPDATE)
            for queue_out in agent.infer_out_queues:
                assert queue_out.get() == SIGN_UPDATED
            agent.get_memory_batch(display.reward_update)

            sample = agent.replay_buffer.sample(1)[0].actions
            if VERBOSE_FIRST and first_collect:
                print("第一次收集的轨迹:")
                for action in sample:
                        print(action_to_str(action))
                first_collect = False
            if not EVAL:
                agent.update(agent.replay_buffer.sample(BATCH_SIZE*N_BATCH_SAMPLE),display.loss_update)
                print("Saving")
                torch.save(agent.policy_model_old.state_dict(), f"{PATH}_policy.pt")
                torch.save(agent.value_model_old.state_dict(), f"{PATH}_value.pt")
                if display.avr_reward_per_trail >= current_max:
                    torch.save(agent.policy_model_old.state_dict(), f"{PATH_MAX}_policy.pt")
                    torch.save(agent.value_model_old.state_dict(), f"{PATH_MAX}_value.pt")
                    current_max = display.avr_reward_per_trail
                with open(REPLAY_BUFFER_FILE,"wb") as file:
                    pickle.dump(agent.replay_buffer, file)
            else:
                sys.exit()
            print("Saved")
            display.reset()
            # 锁是要复位的呢，很难绷忘了一次。
            agent.env_lock = [0]*ENV_NUM
            action_epsilion_scheduler.step()
            n_td_scheduler.step()
        # 多进程关闭
        for queue in agent.env_in_queues:
            queue.put(STOP_SIGN)
        kill_child_processes(os.getpid())
    except (KeyboardInterrupt,SystemExit):
        print("Exit")
    except BaseException:
        Console.print_exception(show_locals=True)
    finally:
        kill_child_processes(os.getpid())
    print(f"Time Usage: {(time.time()-time_start):.4f}")