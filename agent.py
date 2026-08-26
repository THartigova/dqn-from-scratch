import torch 
import torch.nn as nn
from collections import deque
import random
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, stateSize, actionSize):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(stateSize, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, actionSize)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, nextState, done):
        self.buffer.append((state, action, reward, nextState, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, stateSize, actionSize):
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.QNetwork = QNetwork(stateSize, actionSize)
        self.buffer = ReplayBuffer()

        self.epsilon = 1.0
        self.epsilonMin = 0.05
        self.epsilonDecay = 0.995


        self.targetNetwork = QNetwork(stateSize, actionSize)
        self.targetNetwork.load_state_dict(self.QNetwork.state_dict())
        self.optimizer = torch.optim.Adam(self.QNetwork.parameters(), lr=0.001)
        self.gamma = 0.99

    def updateTargetNetwork(self):
        self.targetNetwork.load_state_dict(self.QNetwork.state_dict())

    def selectAction(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionSize - 1)

        stateTensor = torch.FloatTensor(state)
        with torch.no_grad():
            qValues = self.QNetwork(stateTensor)
        return torch.argmax(qValues).item()

    def decayEpsilon(self):
        self.epsilon = max(self.epsilonMin, self.epsilon * self.epsilonDecay)

    def trainStep(self, batch_size = 64):
        if len(self.buffer) < batch_size:
            return None

        batch = self.buffer.sample(batch_size)
        states, actions, rewards, nextStates, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        nextStates = torch.FloatTensor(nextStates)
        dones = torch.FloatTensor(dones)

        currentQ = self.QNetwork(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            nextQ = self.targetNetwork(nextStates).max(1)[0]
            targetQ = rewards + self.gamma * nextQ * (1 - dones)

        loss = nn.functional.mse_loss(currentQ, targetQ)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

if __name__ == "__main__":
    stateSize = 10 * 15
    actionSize = 5
    agent = DQNAgent(stateSize, actionSize)

    for _ in range(200):
        s = [random.random() for _ in range(stateSize)]
        a = random.randint(0, actionSize - 1)
        r = random.random()
        ns = [random.random() for _ in range(stateSize)]
        d = random.random() < 0.1
        agent.buffer.push(s, a, r, ns, d)

    loss = agent.trainStep()
    print(f"Loss po jednom training stepu: {loss}")
