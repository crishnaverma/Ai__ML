import gymnasium as gym
import flappy_bird_gymnasium
import pygame

# Create environment
env = gym.make("FlappyBird-v0", render_mode="human")

# Reset environment
state, info = env.reset()

# Initialize Pygame
pygame.init()

done = False

# Game loop
while not done:

    # Default action: 0 = no flap
    action = 0

    # Check keyboard events
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            done = True

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                action = 1   # flap

    # Take action
    state, reward, terminated, truncated, info = env.step(action)

    # Episode is finished if either happens
    done = terminated or truncated

# Close environment
env.close()
pygame.quit()  