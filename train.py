from game import DodgeGame
from agent import DQNAgent

NUM_EPISODES = 500
TARGET_UPDATE_FREQ = 10
BATCH_SIZE = 64
MAX_STEPS_PER_EPISODE = 1000

game = DodgeGame()
stateSize = game.width * game.height
actionSize = 5
agent = DQNAgent(stateSize, actionSize)

episodeRewards = []
episodeAvgLosses = []
for episode in range(NUM_EPISODES):
    state = game.reset()
    totalReward = 0

    episodeLosses = [] # LOSS
    

    for step in range(MAX_STEPS_PER_EPISODE):
        action = agent.selectAction(state)
        nextState, reward, done, info = game.step(action)

        agent.buffer.push(state, action, reward, nextState, done)
        loss = agent.trainStep(BATCH_SIZE) # pridano LOSS

        if loss is not None:
            episodeLosses.append(loss)

        state = nextState
        totalReward += reward

        if done:
            break

    validLosses = [l for l in episodeLosses if l is not None]
    avgLossThisEpisode = sum(validLosses) / len(validLosses) if validLosses else 0
    episodeAvgLosses.append(avgLossThisEpisode)

    agent.decayEpsilon()
    episodeRewards.append(totalReward)

    if episode % TARGET_UPDATE_FREQ == 0:
        agent.updateTargetNetwork()

    if episode % 10 == 0:
        avgReward = sum(episodeRewards[-10:]) / len(episodeRewards[-10:])
        avgLoss = sum(episodeLosses) /  len(episodeLosses) if episodeLosses else 0
        print(f"Epizoda {episode}: reward={totalReward:.1f}, "
            f"průměr(10)={avgReward:.1f}, epsilon={agent.epsilon:.3f}, "
            f"level={info['level']}, loss={avgLossThisEpisode:.4f}")

import torch
import matplotlib.pyplot as plt

torch.save(agent.QNetwork.state_dict(), "trained_agent.pth")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

ax1.plot(episodeRewards)
ax1.set_xlabel("Epizoda")
ax1.set_ylabel("Celkový reward")
ax1.set_title("Reward v čase")

ax2.plot(episodeAvgLosses)
ax2.set_xlabel("Epizoda")
ax2.set_ylabel("Průměrný loss")
ax2.set_title("Loss v čase")

plt.tight_layout()
plt.savefig("training_progress.png")
plt.show()


