# 一个基于GPT模型的立直麻将bot

本项目实现了一个基于强化学习的立直麻将智能体，使用深度策略梯度方法（如PPO算法）和GPT模型来进行决策学习。

## TODO
 ### 重点
  - 代码解耦和风格优化
  - 修复各个逻辑bug并更新默认超参数
 ### 部署时
  - 添加实际部署的支持(如雀魂)
  - 考虑在第一层估计枚举联合值函数估计?

## 环境需求

- Python 3.11 或更高版本
- PyTorch 2.0 或更高版本
- mahjong 库(需要修复，参阅:https://github.com/MahjongRepository/mahjong/issues/54)

## 运行指导

1. 克隆仓库到本地：
   ```bash
   git clone https://github.com/marko1616/mahjong_DRL
   ```
2. 进入项目目录：
   ```bash
   cd mahjong_DRL
   ```
3. 安装依赖：
   - Torch请自行根据操作系统安装
   ```bash
   pip install -r requirements.txt
   ```

## 使用说明

1. 启动训练：
   ```bash
   python ppo.py
   ```
   这将启动代理的训练过程，训练日志和模型权重会自动保存在指定的目录中(相关超参数调整在各个文件中)。

2. 监控训练：
   使用TensorBoard查看训练进度和性能：

## 项目结构

- `agent.py`: 包含`Agent`类，实现麻将学习代理的主要功能。
- `model.py`: 定义了使用的GPT模型（参考miniGPT）。
- `env.py`: 麻将游戏的环境实现。
- `schedulers.py`: 实现了用于超参数调度的类。
- `ppo.py`: 算法实现，负责启动训练流程。

## 版权和许可

本项目遵循Apache2.0许可证。详细信息请查看`LICENSE`文件。