# 🐍 Snake Game with Responsive Learning

## 🎮 Overview
This project is a **Python implementation** of the classic **Snake game**, enhanced with **Responsive Learning AI** using **PyTorch**. You can play manually or let the AI **learn and control** the snake using a **Deep Q-Network (DQN)**.

---

## 📷 Screenshots  

### 1️⃣ Game Home
![Bank Starting Page](https://raw.githubusercontent.com/karnidipraveen2005/snakeGame/main/screenshots/starting.png)  

### 2️⃣  
![User Sign-Up](https://raw.githubusercontent.com/karnidipraveen2005/snakeGame/main/screenshots/start1.png)  

### 3️⃣ 
![User Login](https://raw.githubusercontent.com/karnidipraveen2005/snakeGame/main/screenshots/start2.png)  

### 4️⃣ Game Over 
![User Home](https://raw.githubusercontent.com/karnidipraveen2005/snakeGame/main/screenshots/ending.png) 

---

## ✨ Features
✅ **Classic Snake Gameplay** – Control the snake to eat food and grow longer.  
🤖 **AI Mode** – Watch the AI learn and play the game.  
🎚 **Multiple Difficulty Levels** – Choose from **Easy, Medium, and Hard**.  
🧠 **Responsive Learning** – The AI improves over time using **Deep Q-Learning**.  

---

## ⚙️ Requirements
📌 **Python 3.x**  
📌 **Pygame**  
📌 **PyTorch**  
📌 **NumPy**  

To install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Installation
Clone the Repository:
```bash
git clone https://github.com/karnidipraveen2005/snakeGame.git
cd snakeGame
```
Install Dependencies:
```bash
pip install -r requirements.txt
```
Run the Game:
```bash
python snake.py
```

---

## 🕹 Main Menu
🔹 **1** – Easy Mode (User plays)  
🔹 **2** – Medium Mode (User plays)  
🔹 **3** – Hard Mode (User plays)  
🔹 **M** – AI Mode (AI plays)  

### 🎮 Game Controls (Manual Mode)
🎯 **Arrow Keys** – Control the direction of the snake.  
❌ **Game Over Options**:
- 🔄 **R** – Restart the game
- ⏹ **ESC** – Exit the game

---

## 🏗 Code Structure
📂 **snake_game.py** – Main script with game logic & AI implementation.  
📜 **README.md** – This file with instructions & project details.  

---

## 🧠 AI Implementation
The AI uses a **Deep Q-Network (DQN)** to learn optimal actions for the snake. It trains using:

### 🔹 State Representation
- **Head position** (x, y)
- **Relative food position**

### 🔹 Action Space
- Move **UP**, **DOWN**, **LEFT**, **RIGHT**

### 🔹 Reward System
- ✅ **+10 points** for eating food.
- ❌ **-1 point** for moving without eating.

### 🔹 Training Details
- **Q-learning with experience replay**
- **Epsilon-greedy exploration**

---

## 🏗 Neural Network Architecture
🟢 **Input Layer** – 4 neurons (head_x, head_y, food_direction_x, food_direction_y)  
🔵 **Hidden Layer 1** – 128 neurons  
🟠 **Hidden Layer 2** – 64 neurons  
🔴 **Output Layer** – 4 neurons (Q-values for each action)  

### 🔹 Hyperparameters
🛠 **Learning Rate** – `0.001`  
🎯 **Discount Factor (Gamma)** – `0.9`  
🔀 **Epsilon (Exploration Rate)** – Starts at `1.0`, decays over time  
📉 **Minimum Epsilon** – `0.01`  
📉 **Epsilon Decay** – `0.995`  

---

## 🤝 Contributing
welcome contributions! 🎉  
To contribute, **fork** the repository and submit a **pull request** with your changes.  

---

## 🌟 Acknowledgments
🔹 **Pygame** – [pygame.org](https://www.pygame.org/)  
🔹 **PyTorch** – [pytorch.org](https://pytorch.org/)  
🔹 **NumPy** – [numpy.org](https://numpy.org/)  

---

## 📩 Contact
📧 **For questions or suggestions**, reach out at **praveenkumar97213@gmail.com**  

🚀 **Happy Coding!** 🐍🎮

