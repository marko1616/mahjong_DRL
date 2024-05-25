import torch
import datetime
from schedulers import Linear_scheduler

# 训练超参数
REPLAY_BUFFER_SIZE = 1000
EPOCHES_PER_UPDATE = 1
ACTION_EPSILION = 0.2
MAX_UPDATE_KL = 1
CLIP_EPSILION = 0.4
WEIGHT_POLICY = 1
WEIGHT_VALUE = 1.5
N_BATCH_SAMPLE = 7
BATCH_SIZE = 100
EPISODES = 175
BTEA = 0.2
LR_VALUE = 0.6e-6
LR_POLICY = 0.6e-6
action_epsilion_scheduler = Linear_scheduler(EPISODES,ACTION_EPSILION,0.001)

# 环境超参数
STABLE_SEED_STEPS = 4# 保持种子在一定时间步内的稳定能增加拟合的可能?也更贴近少初始状态的RL。
INIT_ENV_SEED = 2025
ENV_NUM = 8# 多进程

# 目标超参数
TARGET = "N_step_TD"
LAMBD = 0.1
GAMMA = 0.10
ALPHA = 1
NTD = 2# TD自举步数
alpha_scheduler = Linear_scheduler(EPISODES,ALPHA,0.20)
n_td_scheduler = Linear_scheduler(EPISODES,NTD,1)

# 模型超参数
SEPARATION_LAYER_POLICY = 8
SEPARATION_LAYER_VALUE = 4
DIM_FEEDFORWARD = 1024
ENABLE_COMPILE = False
PAD_TOKEN_ID = 219
MAX_SEQ_LEN = 128
ACTION_SIZE = 185
VOCAB_SIZE = 1 + 184 + 34 + 1# [SEP]，action，手牌，[PAD]
MAX_NORM = 0.5
D_MODEL = 1024# 内部feature维度
DROPOUT = 0.1
NHEAD = 8

# 计算参数&保存参数&显示参数
VERBOSE_POSITIVE_DONE_REWARD = True
REPLAY_BUFFER_FILE = "replay.pkl"
LOG_DIR = f"runs/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
INFER_DEVICES = ["cuda:0"]
DEVICE = "cuda:0"
DTYPE = torch.float
PATH_MAX = "./max"

EVAL = True
if EVAL:
    PATH = PATH_MAX
    MEMGET_NUM_PER_UPDATE = 1000
    ACTION_TEMPERATURE = 0.1
    NDISPLAY = 50
    action_epsilion_scheduler = Linear_scheduler(50,0,0)
else:
    MEMGET_NUM_PER_UPDATE = 600
    ACTION_TEMPERATURE = 0.2
    PATH = "./mahjong"
    NDISPLAY = 10

VERBOSE_FIRST = True
first_collect = True