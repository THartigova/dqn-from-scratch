import torch
import pygame
from game import DodgeGame
from agent import QNetwork

stateSize = 10 * 15
actionSize = 5

qNetwork = QNetwork(stateSize, actionSize)
qNetwork.load_state_dict(torch.load("trained_agent.pth"))
qNetwork.eval()

pygame.init()
game = DodgeGame()
clock = pygame.time.Clock()

state = game.reset()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    stateTensor = torch.FloatTensor(state)
    with torch.no_grad():
        qValues =  qNetwork(stateTensor)
    action = torch.argmax(qValues).item()

    state, reward, done, info = game.step(action)

    if done:
        print(f"Agent prohrál. Level {info['level']}, skóre {info['score']}")
        state = game.reset()

    game.render()
    clock.tick(10)

pygame.quit()
