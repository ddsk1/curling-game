import numpy as np
import gym
from gym import spaces
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import pygame

# 定义全局变量
WIDTH = 100
HEIGHT = 100
BALL_RADIUS = 3
TARGET_RADIUS = 3
TARGET_COLOR = (255, 0, 0)
BALL_COLOR = (0, 0, 255)
BACKGROUND_COLOR = (255, 255, 255)
FPS = 60

# 定义经验回放缓冲区大小和批次大小
REPLAY_MEMORY_SIZE = 10000
BATCH_SIZE = 256

# 定义超参数
GAMMA = 1
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 2000
TARGET_UPDATE = 10
LEARNING_RATE = 1e-3

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义经验回放缓冲区
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# 定义深度Q网络
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CurlingEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        self.ball_radius = 1.0
        self.ball_mass = 1.0
        self.field_width = 100.0
        self.field_height = 100.0
        self.bounce_coefficient = 0.9
        self.control_force = 5.0
        self.air_resistance = 0.005
        self.time_step = 0.1
        self.sub_time_step = 0.01
        self.episode_length = 300

        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.target = np.zeros(2)
        self.time = 0

        self.reset()

    def reset(self):
        self.position = np.random.uniform(low=(self.ball_radius, self.ball_radius),
                                          high=(self.field_width - self.ball_radius, self.field_height - self.ball_radius))
        self.velocity = np.random.uniform(low=-10.0, high=10.0, size=(2,))
        self.target = np.random.uniform(low=(self.ball_radius, self.ball_radius),
                                        high=(self.field_width - self.ball_radius, self.field_height - self.ball_radius))

        self.time = 0

        return self._get_observation()

    def step(self, action):
        force = np.zeros(2)
        if action == 0:  # 左
            force[0] = -self.control_force
        elif action == 1:  # 右
            force[0] = self.control_force
        elif action == 2:  # 上
            force[1] = self.control_force
        elif action == 3:  # 下
            force[1] = -self.control_force
        #elif action == 4:    #保持
            #force = np.zeros(2)

        self._apply_force(force)
        for _ in range(10):
            self._apply_air_resistance()
        self._update_position()
        self._handle_boundary_collision()
        self.time += 1

        distance = np.linalg.norm(self.position - self.target)
        reward = -distance
        done = (self.time >= self.episode_length )

        return self._get_observation(), reward, done

    def render(self):
        pass

    def _get_observation(self):
        return np.concatenate((self.position, self.velocity, self.target))

    def _apply_force(self, force):
        acceleration = force / self.ball_mass
        self.velocity += acceleration * self.time_step

    def _apply_air_resistance(self):
        speed = np.linalg.norm(self.velocity)
        resistance = self.air_resistance * speed * speed
        resistance_force = -resistance * self.velocity / speed
        self.velocity += resistance_force * self.sub_time_step

    def _update_position(self):
        self.position += self.velocity * self.time_step

    def _handle_boundary_collision(self):
        if self.position[0] < self.ball_radius or self.position[0] > self.field_width - self.ball_radius:
            self.velocity[0] *= -self.bounce_coefficient
            self.position[0] = np.clip(self.position[0], self.ball_radius, self.field_width - self.ball_radius)

        if self.position[1] < self.ball_radius or self.position[1] > self.field_height - self.ball_radius:
            self.velocity[1] *= -self.bounce_coefficient
            self.position[1] = np.clip(self.position[1], self.ball_radius, self.field_height - self.ball_radius)
    
    def seed(self, seed=None):
        # Set seed for random number generators
        pass

def train(env, policy_net, target_net, optimizer, memory):
    steps_done = 0
    best_reward = -30000
    rewards = []  # 存储每个episode的奖励值
    losses = []   # 存储每个batch的损失值
    for episode in range(2000):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).view(1, -1)
        episode_reward = 0
        
        for t in range(300):
            # 选择动作
            action = select_action(state, policy_net, steps_done)
            next_state, reward, done = env.step(action)
            episode_reward += reward
            reward = torch.tensor([reward], dtype=torch.float32, device=device)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).view(1, -1)
            
            # 存储经验
            memory.push(state, action, next_state, reward)
            
            # 更新状态
            state = next_state
            
            # 优化模型
            loss = optimize_model(policy_net, target_net, memory, optimizer)
            if loss is not None:
                losses.append(loss.item())
            
            if done:
                break
        
        # 更新目标网络
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # 打印训练信息
        print(f"Episode {episode+1}, Reward: {episode_reward}")

        rewards.append(episode_reward)  # 记录每个episode的奖励值

        if episode_reward > best_reward:
            # 保存模型参数
            torch.save(policy_net.state_dict(), 'best_policy_net.pth')

        
        # 逐渐减小ε
        if steps_done > EPS_DECAY:
            steps_done -= 1
    
    # 绘制训练过程中的奖励变化图
    plot_rewards(rewards)
    
    # 绘制训练过程中的损失变化图
    plot_losses(losses)

# 选择动作
def select_action(state, policy_net, steps_done):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1.0 * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)

# 优化模型
def optimize_model(policy_net, target_net, memory, optimizer):
    if len(memory) < BATCH_SIZE:
        return None
    
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,  dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

# 绘制训练过程中的奖励变化图
def plot_rewards(rewards):
    plt.plot(range(1, len(rewards) + 1), rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward vs Episode')
    plt.savefig('reward_plot.png')  
    plt.show()

# 绘制训练过程中的损失变化图
def plot_losses(losses):
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Loss vs Batch')
    plt.savefig('loss_plot.png')  
    plt.show()

# 创建游戏环境
env = CurlingEnv()

# 创建深度Q网络和目标网络
policy_net = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
target_net = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
target_net.load_state_dict(policy_net.state_dict())

# 定义优化器
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

# 创建经验回放缓冲区
memory = ReplayMemory(REPLAY_MEMORY_SIZE)

# 开始训练
train(env, policy_net, target_net, optimizer, memory)

# 保存模型参数
torch.save(policy_net.state_dict(), 'policy_net.pth')

# 加载模型参数
policy_net = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
policy_net.load_state_dict(torch.load('best_policy_net1.pth'))
policy_net.eval() 

def display_trajectory(env, policy_net):
    # 初始化Pygame
    pygame.init()

    # 创建游戏窗口
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Curling Game")

    # 创建Clock对象来控制帧率
    clock = pygame.time.Clock()

    # 重置环境
    state = env.reset()
    done = False
    reward = -100

    # 渲染游戏界面
    while reward < -5:
        # 绘制游戏界面
        screen.fill(BACKGROUND_COLOR)
        pygame.draw.circle(screen, TARGET_COLOR, (int(env.target[0]), int(env.target[1])), TARGET_RADIUS)
        pygame.draw.circle(screen, BALL_COLOR, (int(env.position[0]), int(env.position[1])), BALL_RADIUS)
        pygame.display.flip()

        # 控制帧率
        clock.tick(FPS)

        # 选择动作
        state = torch.tensor(state, dtype=torch.float32, device=device).view(1, -1)
        with torch.no_grad():
            action = policy_net(state).max(1)[1].view(1, 1).item()

        # 执行动作
        next_state, reward, done = env.step(action)
        state = next_state

    # 退出Pygame
    pygame.quit()

# 显示完整的轨迹
for i in range(10):
    display_trajectory(env, policy_net)
