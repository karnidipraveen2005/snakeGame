import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Initialize Pygame
pygame.init()

# Set up screen
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game with AI")
clock = pygame.time.Clock()

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Snake settings
snake_block_size = 20
snake = [(100, 100)]
direction = "RIGHT"
change_to = direction

# Food position
def generate_food():
    while True:
        food_x = random.randint(0, (WIDTH - snake_block_size) // snake_block_size) * snake_block_size
        food_y = random.randint(0, (HEIGHT - snake_block_size) // snake_block_size) * snake_block_size
        if (food_x, food_y) not in snake:
            return food_x, food_y

food_x, food_y = generate_food()

# Score
score = 0
font = pygame.font.Font(None, 36)

# Reinforcement Learning (DQN) Agent
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# AI Agent
class SnakeAI:
    def __init__(self):
        self.state_size = 6  # [snake_x, snake_y, food_x, food_y, direction_x, direction_y]
        self.action_size = 4  # [UP, DOWN, LEFT, RIGHT]
        self.model = DQN(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = []
    
    def get_state(self):
        head_x, head_y = snake[0]
        state = np.array([head_x, head_y, food_x, food_y, direction == "RIGHT" - direction == "LEFT", direction == "DOWN" - direction == "UP"])
        return torch.tensor(state, dtype=torch.float32)
    
    def get_action(self):
        state = self.get_state()
        with torch.no_grad():
            action_values = self.model(state)
        return torch.argmax(action_values).item()

ai_agent = SnakeAI()
manual_mode = True  # Toggle between manual and AI mode

def draw_snake():
    for x, y in snake:
        pygame.draw.rect(screen, GREEN, (x, y, snake_block_size, snake_block_size))

def draw_food():
    pygame.draw.rect(screen, RED, (food_x, food_y, snake_block_size, snake_block_size))

def show_score():
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))

def check_collision():
    global running
    if snake[0][0] < 0 or snake[0][0] >= WIDTH or snake[0][1] < 0 or snake[0][1] >= HEIGHT:
        running = False
    if snake[0] in snake[1:]:
        running = False

def game_loop(speed):
    global snake, direction, change_to, score, food_x, food_y, running, manual_mode
    snake = [(100, 100)]
    direction = "RIGHT"
    change_to = direction
    score = 0
    food_x, food_y = generate_food()
    running = True

    while running:
        screen.fill(BLACK)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    manual_mode = not manual_mode  # Toggle AI mode
                if manual_mode:
                    if event.key == pygame.K_UP and direction != "DOWN":
                        change_to = "UP"
                    elif event.key == pygame.K_DOWN and direction != "UP":
                        change_to = "DOWN"
                    elif event.key == pygame.K_LEFT and direction != "RIGHT":
                        change_to = "LEFT"
                    elif event.key == pygame.K_RIGHT and direction != "LEFT":
                        change_to = "RIGHT"

        if not manual_mode:
            action = ai_agent.get_action()
            directions = ["UP", "DOWN", "LEFT", "RIGHT"]
            change_to = directions[action]
        
        direction = change_to
        head_x, head_y = snake[0]
        if direction == "UP":
            head_y -= snake_block_size
        elif direction == "DOWN":
            head_y += snake_block_size
        elif direction == "LEFT":
            head_x -= snake_block_size
        elif direction == "RIGHT":
            head_x += snake_block_size

        new_head = (head_x, head_y)
        snake.insert(0, new_head)

        if new_head == (food_x, food_y):
            score += 1
            food_x, food_y = generate_food()
        else:
            snake.pop()

        check_collision()
        draw_snake()
        draw_food()
        show_score()
        pygame.display.update()
        clock.tick(speed)

# Choose difficulty level
def main_menu():
    screen.fill(BLACK)
    title_text = font.render("Choose Difficulty: 1-Easy, 2-Medium, 3-Hard (M for AI Mode)", True, WHITE)
    screen.blit(title_text, (WIDTH//6, HEIGHT//2))
    pygame.display.update()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    game_loop(7)
                elif event.key == pygame.K_2:
                    game_loop(12)
                elif event.key == pygame.K_3:
                    game_loop(18)
                elif event.key == pygame.K_m:
                    global manual_mode
                    manual_mode = False
                    game_loop(10)

main_menu()
