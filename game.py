import random
import pygame
import numpy as np

CELL_SIZE = 40

REWARD_ALIVE = 0.1 
REWARD_BONUS = 5.0 
REWARD_DEATH = -10.0 

COLOR_BG = (18, 18, 24)
COLOR_GRID = (28, 28, 36)
COLOR_PLAYER = (80, 160, 255)
COLOR_PLAYER_GLOW = (40, 90, 160)
COLOR_OBSTACLE = (235, 70, 70)
COLOR_BONUS = (70, 220, 130)

class DodgeGame:
    def __init__(self, width=10, height=15):
        self.width = width
        self.height = height
        self.player_x = width // 2
        self.player_y = height - 1
        self.obstacles = []
        self.level = 1
        self.steps_taken = 0
        self.score = 0
        self.screen = None
        self.bonuses = []

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.width * CELL_SIZE, self.height * CELL_SIZE)
            )
            pygame.display.set_caption("Dodge Game")
            self.font = pygame.font.SysFont("Arial", 20)

        self.screen.fill(COLOR_BG)

        # jemná mřížka na pozadí — pomáhá oku vnímat prostor
        for gx in range(self.width):
            pygame.draw.line(
                self.screen, COLOR_GRID,
                (gx * CELL_SIZE, 0), (gx * CELL_SIZE, self.height * CELL_SIZE)
            )
        for gy in range(self.height):
            pygame.draw.line(
                self.screen, COLOR_GRID,
                (0, gy * CELL_SIZE), (self.width * CELL_SIZE, gy * CELL_SIZE)
            )

        # hráč — zaoblený čtverec s jemnou "glow" vrstvou pod ním
        player_rect = pygame.Rect(
            self.player_x * CELL_SIZE + 2, self.player_y * CELL_SIZE + 2,
            CELL_SIZE - 4, CELL_SIZE - 4
        )
        glow_rect = player_rect.inflate(10, 10)
        pygame.draw.rect(self.screen, COLOR_PLAYER_GLOW, glow_rect, border_radius=12)
        pygame.draw.rect(self.screen, COLOR_PLAYER, player_rect, border_radius=8)

        # překážky — zaoblené, s fading trail nad sebou
        for ox, oy in self.obstacles:
            # trail: 3 stále průhlednější kopie nad překážkou
            for trail_step in range(1, 4):
                trail_y = oy - trail_step
                if trail_y < 0:
                    continue
                alpha = max(0, 90 - trail_step * 30)  # 60, 30, 0
                trail_surface = pygame.Surface((CELL_SIZE - 6, CELL_SIZE - 6), pygame.SRCALPHA)
                pygame.draw.rect(
                    trail_surface, (*COLOR_OBSTACLE, alpha),
                    trail_surface.get_rect(), border_radius=6
                )
                self.screen.blit(trail_surface, (ox * CELL_SIZE + 3, trail_y * CELL_SIZE + 3))

            obs_rect = pygame.Rect(
                ox * CELL_SIZE + 3, oy * CELL_SIZE + 3,
                CELL_SIZE - 6, CELL_SIZE - 6
            )
            pygame.draw.rect(self.screen, COLOR_OBSTACLE, obs_rect, border_radius=6)

        # bonusy — zaoblené, s jemnou září jako u hráče
        for bx, by in self.bonuses:
            bonus_rect = pygame.Rect(
                bx * CELL_SIZE + 5, by * CELL_SIZE + 5,
                CELL_SIZE - 10, CELL_SIZE - 10
            )
            pygame.draw.rect(self.screen, COLOR_BONUS, bonus_rect, border_radius=10)

        score_text = self.font.render(f"Score: {self.score:.0f}", True, (230, 230, 230))
        level_text = self.font.render(f"Level: {self.level}", True, (230, 230, 230))
        self.screen.blit(score_text, (8, 8))
        self.screen.blit(level_text, (8, 30))

        pygame.display.flip()

    def movePlayer(self, action):
        if action == 0:
            self.player_x -= 1

        elif action == 1:
            self.player_x += 1

        elif action == 2:
            self.player_y -= 1

        elif action == 3:
            self.player_y += 1

        self.player_x = max(0, min(self.player_x, self.width - 1))
        self.player_y = max(0, min(self.player_y, self.height - 1))

    def spawnObstacles(self):
        x = random.randint(0, self.width - 1)
        self.obstacles.append([x, 0])

    def moveObstacles(self):
        for obstacle in self.obstacles:
            obstacle[1] += 1
        self.obstacles = [o for o in self.obstacles if o[1] < self.height]

    def checkCollision(self):
        for ox, oy in self.obstacles:
            if ox == self.player_x and oy == self.player_y:
                return True
        return False

    def spawnBonus(self):
        x = random.randint(0, self.width - 1)
        self.bonuses.append([x, 0])

    def moveBonuses(self):
        for bonus in self.bonuses:
            bonus[1] += 1
        self.bonuses = [b for b in self.bonuses if b[1] < self.height]

    def checkBonusPickup(self):
        collected = [b for b in self.bonuses if b[0] == self.player_x and b[1] == self.player_y]
        self.bonuses = [b for b in self.bonuses if b not in collected]
        return len(collected)

    def getState(self):
        grid = np.zeros((self.height, self.width), dtype=np.float32)
        for ox, oy in self.obstacles:
            if 0 <= oy < self.height:
                grid[oy][ox] = -1.0
        for bx, by in self.bonuses:
            if 0 <= by < self.height:
                grid[by][bx] = 1.0
        grid[self.player_y][self.player_x] = 2.0
        return grid.flatten()

    def reset(self):
        self.player_x = self.width // 2
        self.player_y = self.height - 1
        self.obstacles = []
        self.bonuses = []
        self.level = 1
        self.steps_taken = 0
        self.score = 0
        self.particles = []
        return self.getState()

    def step(self, action):
        self.movePlayer(action)
        self.moveObstacles()
        self.moveBonuses()

        obstacle_interval = max(5, 20 - self.level * 2)
        if self.steps_taken % obstacle_interval == 0:
            self.spawnObstacles()

        if self.steps_taken % 35 == 0:
            self.spawnBonus()

        reward = REWARD_ALIVE

        n_collected = self.checkBonusPickup()
        reward += n_collected * REWARD_BONUS
        self.score += n_collected * REWARD_BONUS

        done = self.checkCollision()
        if done:
            reward = REWARD_DEATH
        self.steps_taken += 1
        if self.steps_taken % 200 == 0:
            self.level += 1
        next_state = self.getState()
        info = {"level": self.level, "score": self.score}
        return next_state, reward, done, info

if __name__ == "__main__":
    import random as rnd

    pygame.init()

    game = DodgeGame()
    clock = pygame.time.Clock()
    running = True
    tick = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        action = 4
        if keys[pygame.K_LEFT]:
            game.movePlayer(0)
        elif keys[pygame.K_RIGHT]:
            game.movePlayer(1)
        elif keys[pygame.K_UP]:
            game.movePlayer(2)
        elif keys[pygame.K_DOWN]:
            game.movePlayer(3)

        state, reward, done, info = game.step(action)

        if done:
            print(f"Konec hry! Level {info['level']}, skóre {info['score']}")
            running = False

        game.render()
        clock.tick(10)
        tick += 1
    pygame.quit()