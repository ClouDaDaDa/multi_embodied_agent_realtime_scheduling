# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np
# import pygame
#
#
# class MyEnv(gym.Env):
#     metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
#
#     def __init__(self, render_mode=None):
#         super(MyEnv, self).__init__()
#
#         # Define action and observation spaces
#         self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)  # 2D velocity control
#         self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(4,), dtype=np.float32)  # [x, y, vx, vy]
#
#         # Environment parameters
#         self.max_steps = 200
#         self.current_step = 0
#         self.position = np.zeros(2)  # [x, y]
#         self.velocity = np.zeros(2)  # [vx, vy]
#         self.target = np.array([5.0, 5.0])  # Target position
#
#         # Rendering settings
#         self.render_mode = render_mode
#         self.screen_size = (800, 600)  # Window size
#         self.scale = 40  # Pixels per unit
#         self.screen = None
#         self.clock = None
#         self.surface = None
#
#         # Initialize rendering if needed
#         if self.render_mode in ["human", "rgb_array"]:
#             pygame.init()
#             if self.render_mode == "human":
#                 self.screen = pygame.display.set_mode(self.screen_size)
#                 pygame.display.set_caption("MyEnv Visualization")
#             self.clock = pygame.time.Clock()
#             self.surface = pygame.Surface(self.screen_size)
#
#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         self.current_step = 0
#         self.position = np.zeros(2)
#         self.velocity = np.zeros(2)
#         observation = np.concatenate([self.position, self.velocity])
#         return observation, {}
#
#     def step(self, action):
#         self.current_step += 1
#
#         # Update velocity and position
#         self.velocity = np.clip(action, -1.0, 1.0)
#         self.position += self.velocity * 0.1  # Simple physics: dt = 0.1
#         self.position = np.clip(self.position, -10.0, 10.0)
#
#         # Calculate reward
#         distance = np.linalg.norm(self.position - self.target)
#         reward = -distance
#
#         # Check termination
#         terminated = distance < 0.5  # Reach target
#         truncated = self.current_step >= self.max_steps
#
#         observation = np.concatenate([self.position, self.velocity])
#         return observation, reward, terminated, truncated, {}
#
#     def render(self):
#         if self.render_mode is None:
#             gym.logger.warn("You are calling render method without specifying any render mode.")
#             return None
#
#         if self.render_mode not in ["human", "rgb_array"]:
#             raise NotImplementedError(f"Render mode '{self.render_mode}' is not supported.")
#
#         # Clear surface
#         self.surface.fill((255, 255, 255))  # White background
#
#         # Draw target (red circle)
#         target_pos = (int(self.target[0] * self.scale + self.screen_size[0] // 2),
#                       int(self.target[1] * self.scale + self.screen_size[1] // 2))
#         pygame.draw.circle(self.surface, (255, 0, 0), target_pos, 10)
#
#         # Draw agent (blue rectangle)
#         agent_pos = (int(self.position[0] * self.scale + self.screen_size[0] // 2),
#                      int(self.position[1] * self.scale + self.screen_size[1] // 2))
#         pygame.draw.rect(self.surface, (0, 0, 255), (*agent_pos, 20, 20))
#
#         # Draw info (step count)
#         font = pygame.font.SysFont(None, 36)
#         text = font.render(f"Step: {self.current_step}", True, (0, 0, 0))
#         self.surface.blit(text, (10, 10))
#
#         if self.render_mode == "human":
#             # Update display
#             self.screen.blit(self.surface, (0, 0))
#             pygame.display.flip()
#             self.clock.tick(self.metadata["render_fps"])
#         elif self.render_mode == "rgb_array":
#             # Convert surface to RGB array
#             return np.transpose(pygame.surfarray.array3d(self.surface), axes=(1, 0, 2))
#
#     def close(self):
#         if self.screen is not None:
#             pygame.display.quit()
#             pygame.quit()
#             self.screen = None
#
# # Test the environment
# def test_environment():
#     env = MyEnv(render_mode="human")
#     observation, _ = env.reset()
#     done = False
#
#     while not done:
#         action = env.action_space.sample()  # Random action
#         observation, reward, terminated, truncated, _ = env.step(action)
#         env.render()  # Render after each step
#         done = terminated or truncated
#
#         # Handle Pygame events (e.g., close window)
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 done = True
#
#     env.close()
#
#
# if __name__ == "__main__":
#     test_environment()
#
#

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class MyEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None):
        super(MyEnv, self).__init__()

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(4,), dtype=np.float32)

        # Environment parameters
        self.max_steps = 200
        self.current_step = 0
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.target = np.array([5.0, 5.0])

        # Rendering settings
        self.render_mode = render_mode
        self.fig = None
        self.ax = None
        self.agent_plot = None
        self.target_plot = None
        self.text = None

        # Initialize rendering if needed
        if self.render_mode in ["human", "rgb_array"]:
            plt.ion()  # Enable interactive mode
            self.fig, self.ax = plt.subplots(figsize=(8, 6))
            self.ax.set_xlim(-10, 10)
            self.ax.set_ylim(-10, 10)
            self.ax.set_xlabel("X Position")
            self.ax.set_ylabel("Y Position")
            self.ax.grid(True)

            # Initial empty plots
            self.agent_plot, = self.ax.plot([], [], 'bo', markersize=10, label='Agent')
            self.target_plot, = self.ax.plot(self.target[0], self.target[1], 'ro', markersize=10, label='Target')
            self.text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, fontsize=12)
            self.ax.legend()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        observation = np.concatenate([self.position, self.velocity])
        return observation, {}

    def step(self, action):
        self.current_step += 1

        # Update velocity and position
        self.velocity = np.clip(action, -1.0, 1.0)
        self.position += self.velocity * 0.1
        self.position = np.clip(self.position, -10.0, 10.0)

        # Calculate reward
        distance = np.linalg.norm(self.position - self.target)
        reward = -distance

        # Check termination
        terminated = distance < 0.5
        truncated = self.current_step >= self.max_steps

        observation = np.concatenate([self.position, self.velocity])
        return observation, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn("You are calling render method without specifying any render mode.")
            return None

        if self.render_mode not in ["human", "rgb_array"]:
            raise NotImplementedError(f"Render mode '{self.render_mode}' is not supported.")

        # Update agent position
        self.agent_plot.set_data([self.position[0]], [self.position[1]])

        # Update step count text
        self.text.set_text(f"Step: {self.current_step}")

        if self.render_mode == "human":
            # Redraw and pause for real-time display
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(1 / self.metadata["render_fps"])

        elif self.render_mode == "rgb_array":
            # Convert figure to RGB array
            self.fig.canvas.draw()
            width, height = self.fig.canvas.get_width_height()
            buffer = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            buffer = buffer.reshape((height, width, 3))
            return buffer

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            plt.ioff()

# Test the environment
def test_environment():
    env = MyEnv(
        # render_mode="human"
        render_mode="rgb_array"
    )
    observation, _ = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, _ = env.step(action)
        env.render()
        done = terminated or truncated

    env.close()

if __name__ == "__main__":
    test_environment()


