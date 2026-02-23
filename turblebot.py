import pygame
import math
from math import sin, cos, pi
import time
import random
import numpy as np
import threading
import matplotlib.pyplot as plt
from collections import deque

class turtlebot:
    def __init__(self, name):
        # robot specifications
        self.name = name
        self.length = 138  # millimeters
        self.width = 178   # millimeters
        self.height = 192  # millimeters
        self.radius = 40   # millimeters
        self.wheel_radius = 33  # millimeters
        self.wheel_width = 18   # millimeters
        self.mass = 1.0    # kg

        # initial pose
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    def get_specs(self):
        return {
            "name": self.name,
            "length": self.length,
            "width": self.width,
            "height": self.height,
            "radius": self.radius,
            "wheel_radius": self.wheel_radius,
            "wheel_width": self.wheel_width,
            "mass": self.mass
        }
    
    def joint_states(self):
        # 20 Hz update rate
        return {
            "left_wheel_position": 0.0,
            "right_wheel_position": 0.0,
            "left_wheel_velocity": 0.0,
            "right_wheel_velocity": 0.0
        }
    
    def imu_data(self):
        # 20 Hz update rate
        return {
            "orientation": (0.0, 0.0, 0.0, 1.0),
            "angular_velocity": (0.0, 0.0, 0.0),
            "linear_acceleration": (0.0, 0.0, 9.81)
        }
    
    
    
    def scan_data(self):
        # 5 Hz update rate
        return [float('inf')] * 360  # Simulated LIDAR data with no obstacles
    
    def differential_drive(self, left_wheel_velocity, right_wheel_velocity):
        # Simple differential drive kinematics
        linear_velocity = (left_wheel_velocity + right_wheel_velocity) / 2.0
        angular_velocity = (right_wheel_velocity - left_wheel_velocity) / (2 * self.radius)
        return linear_velocity, angular_velocity

    def update_pose(self, linear_velocity, angular_velocity, duration):
        # 20 Hz update rate
        delta_theta = angular_velocity * duration
        delta_x = linear_velocity * -sin(self.theta + delta_theta / 2) * duration
        delta_y = linear_velocity * -cos(self.theta + delta_theta / 2) * duration
        self.x += delta_x
        self.y += delta_y
        self.theta += delta_theta
    
    def ekf_odometry(self):
        # Placeholder for EKF odometry
        return {
            "x": self.x,
            "y": self.y,
            "theta": self.theta
        }


class TurtlebotSimulator:
    def __init__(self, turtlebot):
        self.turtlebot = turtlebot
        pygame.init()
        self.screen = pygame.display.set_mode((1500, 1000))
        pygame.display.set_caption(f"{self.turtlebot.name} Simulator")
        self.clock = pygame.time.Clock()
        self.running = True
        # Maze walls (x, y, width, height)
        self.walls = [
            (100, 100, 1200, 20),   # Top wall
            (100, 100, 20, 800),    # Left wall
            (100, 880, 1200, 20),   # Bottom wall
            (1280, 100, 20, 800),   # Right wall

            # Inner maze structure
            (300, 100, 20, 600),
            (500, 300, 600, 20),
            (700, 100, 20, 500),
            (900, 500, 300, 20),
            (400, 700, 600, 20)
        ]

        # Goal position
        self.goal = (1200, 800)
        self.goal_radius = 40    
        # --- Spawn point (start pose) ---
        self.spawn = (160, 160, 0.0)  # (x, y, theta) inside the outer walls
        # Safe spawn point (must be far from walls by at least radius + a little margin)
        margin = self.turtlebot.radius + 30
        self.turtlebot.x = 100 + 20 + margin   # left wall x + thickness + margin
        self.turtlebot.y = 100 + 20 + margin   # top wall y + thickness + margin
        self.turtlebot.theta = 0.0

    def draw_maze(self):
        # Draw walls
        for wall in self.walls:
            pygame.draw.rect(self.screen, (0, 0, 0), wall)

        # Draw goal
        pygame.draw.circle(self.screen, (0, 255, 0), self.goal, self.goal_radius)

    def check_collision(self):
        robot_rect = pygame.Rect(
            self.turtlebot.x - self.turtlebot.radius,
            self.turtlebot.y - self.turtlebot.radius,
            self.turtlebot.radius * 2,
            self.turtlebot.radius * 2
        )

        for wall in self.walls:
            if robot_rect.colliderect(pygame.Rect(wall)):
                return True
        return False

    def check_goal(self):
        dx = self.turtlebot.x - self.goal[0]
        dy = self.turtlebot.y - self.goal[1]
        distance = math.sqrt(dx**2 + dy**2)

        if distance < self.goal_radius:
            print("🎯 GOAL REACHED!")
            self.running = False

    def turtlebot_specs(self):
        return self.turtlebot.get_specs()

    def turtlebot_draw(self):
        x = int(self.turtlebot.x)
        y = int(self.turtlebot.y)
        r = int(self.turtlebot.radius)

        # Body
        pygame.draw.circle(self.screen, (0, 0, 255), (x, y), r)

        # Wheel geometry (scaled)
        wheel_w = int(self.turtlebot.wheel_width)
        wheel_h = int(self.turtlebot.wheel_radius * 2)

        # Put wheels slightly outside the body
        wheel_offset_x = r + 10
        wheel_offset_y = wheel_h // 2

        # Left wheel
        pygame.draw.rect(
            self.screen, (255, 0, 0),
            (x - wheel_offset_x - wheel_w, y - wheel_offset_y, wheel_w, wheel_h),
            0
        )

        # Right wheel
        pygame.draw.rect(
            self.screen, (255, 0, 0),
            (x + wheel_offset_x, y - wheel_offset_y, wheel_w, wheel_h),
            0
        )

        # Heading
        heading_x = self.turtlebot.x + r * -sin(self.turtlebot.theta)
        heading_y = self.turtlebot.y + r * -cos(self.turtlebot.theta)
        pygame.draw.line(self.screen, (0, 255, 0), (x, y), (int(heading_x), int(heading_y)), 5)

    def keyboard_input(self):
        keys = pygame.key.get_pressed()
        left_wheel_velocity = 0.0
        right_wheel_velocity = 0.0
        if keys[pygame.K_UP]:
            left_wheel_velocity += 5.0
            right_wheel_velocity += 5.0
        if keys[pygame.K_DOWN]:
            left_wheel_velocity -= 5.0
            right_wheel_velocity -= 5.0
        if keys[pygame.K_LEFT]:
            left_wheel_velocity -= 2.50
            right_wheel_velocity += 2.50
        if keys[pygame.K_RIGHT]:
            left_wheel_velocity += 2.50
            right_wheel_velocity -= 2.50
        return left_wheel_velocity, right_wheel_velocity
    
    def start_plot_thread(self):
        # time-series buffers
        self.t_buf = deque(maxlen=4000)
        self.x_buf = deque(maxlen=4000)
        self.y_buf = deque(maxlen=4000)
        self.v_buf = deque(maxlen=4000)     # linear velocity
        self.w_buf = deque(maxlen=4000)     # angular velocity

        self.t0 = time.time()
        self.plot_running = True
        self.plot_thread = threading.Thread(target=self._plot_loop_timeseries, daemon=True)
        self.plot_thread.start()

    def _plot_loop_timeseries(self):
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_title("Turtlebot Time Series (Real-time)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Value")

        line_v, = ax.plot([], [], label="Linear Velocity")
        line_w, = ax.plot([], [], label="Angular Velocity")
        ax.legend()

        while self.plot_running:
            if len(self.t_buf) > 1:
                t = [t - self.t0 for t in self.t_buf]
                line_v.set_data(t, self.v_buf)
                line_w.set_data(t, self.w_buf)

                # auto-follow view (optional)
                ax.set_xlim(min(t) - 1, max(t) + 1)
                ax.set_ylim(min(self.v_buf + self.w_buf) - 10, max(self.v_buf + self.w_buf) + 10)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            time.sleep(0.05)  # ~20 Hz plot update

    def _plot_loop_timeseries(self):
        plt.ion()

        # Figure 1: Position vs time
        fig_pos, ax_pos = plt.subplots()
        ax_pos.set_title("Position vs Time")
        ax_pos.set_xlabel("time (s)")
        ax_pos.set_ylabel("position (mm)")
        ax_pos.grid(True)
        line_x, = ax_pos.plot([], [], '-')  # x(t)
        line_y, = ax_pos.plot([], [], '-')  # y(t)
        ax_pos.legend([line_x, line_y], ["x(t)", "y(t)"])

        # Figure 2: Velocity vs time
        fig_vel, ax_vel = plt.subplots()
        ax_vel.set_title("Velocity vs Time")
        ax_vel.set_xlabel("time (s)")
        ax_vel.set_ylabel("velocity")
        ax_vel.grid(True)
        line_v, = ax_vel.plot([], [], '-')  # v(t)
        line_w, = ax_vel.plot([], [], '-')  # w(t)
        ax_vel.legend([line_v, line_w], ["v(t) mm/s", "ω(t) rad/s"])

        # Fixed x-window to avoid expensive autoscale every update
        window_s = 10.0

        while self.plot_running:
            if len(self.t_buf) > 5:
                t = list(self.t_buf)
                x = list(self.x_buf)
                y = list(self.y_buf)
                v = list(self.v_buf)
                w = list(self.w_buf)

                # update data
                line_x.set_data(t, x)
                line_y.set_data(t, y)
                line_v.set_data(t, v)
                line_w.set_data(t, w)

                # show last N seconds for speed
                t_max = t[-1]
                t_min = max(0.0, t_max - window_s)

                ax_pos.set_xlim(t_min, t_max)
                ax_vel.set_xlim(t_min, t_max)

                # y autoscale cheaply (based on last window only)
                # find indices in window
                k0 = 0
                for k in range(len(t)-1, -1, -1):
                    if t[k] < t_min:
                        k0 = k
                        break

                xs = x[k0:]
                ys = y[k0:]
                vs = v[k0:]
                ws = w[k0:]

                ax_pos.set_ylim(min(xs + ys) - 50, max(xs + ys) + 50)
                ax_vel.set_ylim(min(vs + ws) - 1, max(vs + ws) + 1)

                fig_pos.canvas.draw_idle()
                fig_vel.canvas.draw_idle()
                fig_pos.canvas.flush_events()
                fig_vel.canvas.flush_events()

            time.sleep(0.1)  # 10 Hz plot update (lighter)

        plt.close(fig_pos)
        plt.close(fig_vel)

    def run(self):
        # Safe spawn
        margin = self.turtlebot.radius + 30
        self.turtlebot.x = 100 + 20 + margin
        self.turtlebot.y = 100 + 20 + margin
        self.turtlebot.theta = 0.0

        self.start_plot_thread()

        physics_dt = 0.005        # 20 Hz physics
        render_fps = 60          # smooth input/display
        acc = 0.0
        last = time.time()

        while self.running:
            now = time.time()
            frame_dt = now - last
            last = now
            acc += frame_dt

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            # Read keyboard every frame (responsive)
            left_wheel_velocity, right_wheel_velocity = self.keyboard_input()
            linear_velocity, angular_velocity = self.turtlebot.differential_drive(left_wheel_velocity, right_wheel_velocity)

            # Physics update at fixed rate
            while acc >= physics_dt:
                self.turtlebot.update_pose(linear_velocity, angular_velocity, physics_dt)
                acc -= physics_dt

                # log time-series once per physics step
                t = time.time() - self.t0
                self.t_buf.append(t)
                self.x_buf.append(self.turtlebot.x)
                self.y_buf.append(self.turtlebot.y)
                self.v_buf.append(linear_velocity)
                self.w_buf.append(angular_velocity)

                if self.check_collision():
                    print("💥 Collision!")
                    self.running = False
                    break

                self.check_goal()

            # Render at high FPS
            self.screen.fill((255, 255, 255))
            self.draw_maze()
            self.turtlebot_draw()
            pygame.display.flip()
            self.clock.tick(render_fps)

        self.plot_running = False
        pygame.quit()

if __name__ == "__main__":
    my_turtlebot = turtlebot("Turtlebot3")
    simulator = TurtlebotSimulator(my_turtlebot)
    simulator.run()