import os
import gym
import numpy as np
import collections
import cv2
import retro
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import random
import multiprocessing


class BasicWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.actions = np.array([
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 0]
        ])
        self.action_space = gym.spaces.Discrete(3)

    def step(self, action):
        action = self.actions[action]
        next_state, reward, done, info = self.env.step(action)
        if info['lives'] < 2:
            done = True
        return next_state, reward, done, info

    def reset(self):
        return super().reset()


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init to first obs"""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    """
    Downsamples image to 84x84
    Greyscales image

    Returns numpy array
    """

    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 224 * 240 * 3:
            img = np.reshape(frame, [224, 240, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    """Normalize pixel values in frame --> 0 to 1"""

    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


def make_env(env):
    env = BasicWrapper(env)
    env = MaxAndSkipEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    env = ScaledFloatFrame(env)
    return env


class ReplayMemory(object):
    def __init__(self, max_memory_size, batch_size, state_space):
        self.max_memory_size = max_memory_size
        self.batch_size = batch_size
        self.memory = collections.deque(maxlen=max_memory_size)
        self.obs_ = torch.zeros((batch_size, *state_space), dtype=torch.float32)
        self.obs_next_ = torch.zeros((batch_size, *state_space), dtype=torch.float32)
        self.actions_ = torch.zeros((batch_size, 1), dtype=torch.int64)
        self.rewards_ = torch.zeros((batch_size, 1), dtype=torch.float32)
        self.dones_ = torch.zeros((batch_size, 1), dtype=torch.float32)

    def sample(self):
        idx = np.random.choice(len(self.memory), self.batch_size, replace=False)
        for i in range(self.batch_size):
            self.obs_[i] = torch.tensor(self.memory[idx[i]][0], dtype=torch.float32)
            self.actions_[i] = torch.tensor(self.memory[idx[i]][1], dtype=torch.int64)
            self.rewards_[i] = torch.tensor(self.memory[idx[i]][2], dtype=torch.float32)
            self.obs_next_[i] = torch.tensor(self.memory[idx[i]][3], dtype=torch.float32)
            self.dones_[i] = torch.tensor(self.memory[idx[i]][4], dtype=torch.float32)
        return self.obs_, self.actions_, self.rewards_, self.obs_next_, self.dones_

    def remember(self, trans):
        self.memory.append(trans)

    def current_size(self):
        return len(self.memory)


class DQNSolver(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(DQNSolver, self).__init__()
        self.cnn1 = nn.Conv2d(4, 64, kernel_size=3, stride=3)
        self.cnn2 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.cnn3 = nn.Conv2d(128, 32, kernel_size=3, stride=1)
        self.pooling = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(512, 1024)
        self.linear2 = nn.Linear(1024, 3)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.pooling(x)
        x = self.cnn2(x)
        x = self.pooling(x)
        x = self.cnn3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class Agent(object):
    def __init__(self, max_memory_size, batch_size, state_space, action_space):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.state_space = state_space
        self.max_memory_size = max_memory_size
        self.memory = ReplayMemory(max_memory_size, batch_size, state_space)
        self.predict_net = DQNSolver(state_space, action_space).to(self.device)
        self.target_net = DQNSolver(state_space, action_space).to(self.device)
        self.optimizer = torch.optim.Adam(self.predict_net.parameters(), lr=25e-5)
        self.copy = 5000
        self.step = 0

        self.gamma = 0.9
        self.exploration_max = 1.0
        self.exploration_rate = 1.0
        self.exploration_min = 0.02
        self.exploration_decay = 0.99
        self.action_space = action_space

    def act(self, state):
        self.step += 1
        state = torch.tensor([state])
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.action_space, 1)[0]

        return int(torch.argmax(self.predict_net(state.to(self.device))).cpu())

    def copy_model(self):
        self.target_net.load_state_dict(self.predict_net.state_dict())

    def remember(self, trans):
        return self.memory.remember(trans)

    def update(self):
        if self.step % self.copy == 0:
            self.copy_model()

        if self.memory.current_size() < self.memory.batch_size:
            return
        self.optimizer.zero_grad()
        STATE, ACTION, REWARD, STATE2, DONE = self.memory.sample()
        STATE = STATE.to(self.device)
        ACTION = ACTION.to(self.device)
        REWARD = REWARD.to(self.device)
        STATE2 = STATE2.to(self.device)
        DONE = DONE.to(self.device)

        q_values = self.predict_net(STATE)
        prediction = torch.gather(q_values, dim=1, index=ACTION)

        q_values_next = self.target_net(STATE2)
        target = REWARD + self.gamma * torch.mul(torch.max(q_values_next, dim=1, keepdim=True).values, 1 - DONE)

        loss = F.smooth_l1_loss(prediction, target)

        loss.backward()
        self.optimizer.step()
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_rate, self.exploration_min)


def play(agent, update_lock, total_episodes, rewards, seed):
    pid = os.getpid()
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = retro.make('SuperMarioBros-Nes', state='Level1-1')
    env = make_env(env)  # Wraps the environment so that frames are grayscale

    for ep_num in range(2500):
        state = env.reset()
        total_reward = 0
        steps = 0
        terminal = False
        while not terminal:
            action = agent.act(state)
            steps += 1
            state_next, reward, terminal, info = env.step(action)
            total_reward += reward
            trans = (state, action, reward, state_next, terminal)
            update_lock.acquire()
            agent.remember(trans)
            update_lock.release()

            update_lock.acquire()
            agent.update()
            update_lock.release()

            state = state_next

        rewards.append(total_reward)
        logger.info('PID:{}, Episode:{}, Total Steps: {}, Total Reward:{}'.format(pid, total_episodes.value, steps, total_reward))
        total_episodes.value += 1

        if total_episodes.value % 100 == 99:
            logger.info('Last 100 episodes, average reward:{}'.format(np.mean(rewards[-100:])))
    env.close()


class MyManager(multiprocessing.managers.BaseManager):
    def __init__(self):
        super(MyManager, self).__init__()


MyManager.register('Memory', ReplayMemory)

MyManager.register('Value', multiprocessing.Value)

MyManager.register('Agent', Agent)

MyManager.register('list', list, multiprocessing.managers.ListProxy)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("Mario_Training")
    state_space = (4, 84, 84)
    workers = 4
    with MyManager() as manager:
        agent = manager.Agent(max_memory_size=30000, batch_size=32, state_space=state_space, action_space=3)
        total_episodes = multiprocessing.Value('i', 0)
        rewards = manager.list()
        update_lock = multiprocessing.Lock()
        processes = []
        seeds = [np.random.randint(1000) for _ in range(workers)]
        print(seeds)
        for seed in seeds:
            p = multiprocessing.Process(target=play, args=(agent, update_lock, total_episodes, rewards, seed))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
