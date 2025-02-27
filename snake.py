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
pygame.display.set_caption("Snake Game with Responsive Learning")
clock = pygame.time.Clock()

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)

# Snake settings
snake_block_size = 20
snake = [(100, 100)]
direction = "RIGHT"
change_to = direction

# Food position
def generate_food():
    while True:
        food_x = random.randint(1, (WIDTH - snake_block_size) // snake_block_size - 1) * snake_block_size
        food_y = random.randint(1, (HEIGHT - snake_block_size) // snake_block_size - 1) * snake_block_size
        if (food_x, food_y) not in snake:
            return food_x, food_y

food_x, food_y = generate_food()

# Score
score = 0
font = pygame.font.Font(None, 36)

# PyTorch Neural Network
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Hyperparameters
INPUT_SIZE = 4  # head_x, head_y, food_direction_x, food_direction_y
OUTPUT_SIZE = 4  # UP, DOWN, LEFT, RIGHT
LEARNING_RATE = 0.001
GAMMA = 0.9
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01

# Initialize model, optimizer, and loss function
model = DQN(INPUT_SIZE, OUTPUT_SIZE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

# Helper functions for PyTorch AI
def get_state():
    head_x, head_y = snake[0]
    food_direction_x = 1 if food_x > head_x else -1 if food_x < head_x else 0
    food_direction_y = 1 if food_y > head_y else -1 if food_y < head_y else 0
    return np.array([head_x, head_y, food_direction_x, food_direction_y], dtype=np.float32)

def choose_action(state):
    if random.uniform(0, 1) < EPSILON:
        return random.randint(0, 3)  # Explore: random action
    else:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32)
            q_values = model(state_tensor)
            return torch.argmax(q_values).item()  # Exploit: best action

def train_model(state, action, reward, next_state, done):
    state_tensor = torch.tensor(state, dtype=torch.float32)
    next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
    action_tensor = torch.tensor(action, dtype=torch.int64)
    reward_tensor = torch.tensor(reward, dtype=torch.float32)

    # Current Q-value
    q_values = model(state_tensor)
    current_q = q_values[action_tensor]

    # Target Q-value
    with torch.no_grad():
        next_q_values = model(next_state_tensor)
        max_next_q = torch.max(next_q_values)
        target_q = reward_tensor + (1 - done) * GAMMA * max_next_q

    # Loss and optimization
    loss = loss_fn(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Game functions
def draw_snake():
    for x, y in snake:
        pygame.draw.rect(screen, GREEN, (x, y, snake_block_size, snake_block_size))

def draw_food():
    pygame.draw.rect(screen, YELLOW, (food_x, food_y, snake_block_size, snake_block_size))

def draw_walls():
    for x in range(0, WIDTH, snake_block_size):
        pygame.draw.rect(screen, RED, (x, 0, snake_block_size, snake_block_size))
        pygame.draw.rect(screen, RED, (x, HEIGHT - snake_block_size, snake_block_size, snake_block_size))
    for y in range(0, HEIGHT, snake_block_size):
        pygame.draw.rect(screen, RED, (0, y, snake_block_size, snake_block_size))
        pygame.draw.rect(screen, RED, (WIDTH - snake_block_size, y, snake_block_size, snake_block_size))

def show_score():
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))

def check_collision():
    global running
    head_x, head_y = snake[0]
    if head_x <= 0 or head_x >= WIDTH - snake_block_size or head_y <= 0 or head_y >= HEIGHT - snake_block_size:
        running = False
    if snake[0] in snake[1:]:
        running = False

def game_over():
    screen.fill(CYAN)
    game_over_text = font.render("Game Over! Press R to Restart or ESC to Exit", True, WHITE)
    screen.blit(game_over_text, (WIDTH // 6, HEIGHT // 2))
    pygame.display.update()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    main_menu()
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return

def game_loop(speed, ai_mode):
    global snake, direction, change_to, score, food_x, food_y, running, EPSILON
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
                if not ai_mode:
                    if event.key == pygame.K_UP and direction != "DOWN":
                        change_to = "UP"
                    elif event.key == pygame.K_DOWN and direction != "UP":
                        change_to = "DOWN"
                    elif event.key == pygame.K_LEFT and direction != "RIGHT":
                        change_to = "LEFT"
                    elif event.key == pygame.K_RIGHT and direction != "LEFT":
                        change_to = "RIGHT"

        state = get_state()
        if ai_mode:
            action = choose_action(state)
            if action == 0:
                change_to = "UP"
            elif action == 1:
                change_to = "DOWN"
            elif action == 2:
                change_to = "LEFT"
            elif action == 3:
                change_to = "RIGHT"

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
            reward = 10  # Reward for eating food
        else:
            snake.pop()
            reward = -1  # Penalty for moving without food

        next_state = get_state()
        done = not running

        if ai_mode:
            train_model(state, action, reward, next_state, done)
            EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)  # Decay epsilon

        check_collision()
        draw_walls()
        draw_snake()
        draw_food()
        show_score()
        pygame.display.update()
        clock.tick(speed)

    # Call game_over() when the game ends
    game_over()

def main_menu():
    screen.fill(BLACK)
    title_text = font.render("Select Level: 1-Easy, 2-Medium, 3-Hard, M-AI Mode", True, WHITE)
    screen.blit(title_text, (WIDTH // 6, HEIGHT // 2))
    pygame.display.update()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    game_loop(7, False)  # Easy (user plays)
                elif event.key == pygame.K_2:
                    game_loop(12, False)  # Medium (user plays)
                elif event.key == pygame.K_3:
                    game_loop(18, False)  # Hard (user plays)
                elif event.key == pygame.K_m:
                    game_loop(10, True)  # AI Mode (AI plays)

main_menu() 