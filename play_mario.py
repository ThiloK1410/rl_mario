#!/usr/bin/env python3
"""
Mario Bros Human Player Script
Play Super Mario Bros using keyboard controls.
Uses the same environment as the RL training but simplified for human play.
"""

import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import keyboard
import time
import sys
import threading
import os
import numpy as np
import json
from datetime import datetime

def create_human_env():
    """Create a Mario environment optimized for human play."""
    # Use the same base environment as training but without heavy wrappers
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    return env

class KeyboardController:
    """Handle keyboard input for Mario controls."""
    
    def __init__(self):
        self.current_action = 0
        self.quit_game = False
        self.reset_requested = False
        self.save_requested = False
        self.action_meanings = [
            'NOOP',           # 0
            'RIGHT',          # 1  
            'RIGHT + A',      # 2
            'RIGHT + B',      # 3
            'RIGHT + A + B',  # 4
            'A',              # 5
            'LEFT',           # 6
            'LEFT + A',       # 7
            'LEFT + B',       # 8
            'LEFT + A + B',   # 9
            'DOWN',           # 10
            'UP'              # 11
        ]
        
    def get_action(self):
        """Get current action based on pressed keys."""
        if self.quit_game:
            return -1
        
        # Check for special commands first (with debouncing)
        if keyboard.is_pressed('f5'):  # Reset
            if not self.reset_requested:
                self.reset_requested = True
                return -2  # Special reset action
        else:
            self.reset_requested = False
            
        if keyboard.is_pressed('f2'):  # Save state
            if not self.save_requested:
                self.save_requested = True
                return -3  # Special save action
        else:
            self.save_requested = False
            
        # Check key combinations in priority order
        right = keyboard.is_pressed('right')
        left = keyboard.is_pressed('left')
        up = keyboard.is_pressed('up')
        down = keyboard.is_pressed('down')
        a_button = keyboard.is_pressed('space')  # Jump
        b_button = keyboard.is_pressed('shift')  # Run/Fire
        
        # Map key combinations to actions
        if right and a_button and b_button:
            return 4  # RIGHT + A + B
        elif right and a_button:
            return 2  # RIGHT + A
        elif right and b_button:
            return 3  # RIGHT + B
        elif right:
            return 1  # RIGHT
        elif left and a_button and b_button:
            return 9  # LEFT + A + B
        elif left and a_button:
            return 7  # LEFT + A
        elif left and b_button:
            return 8  # LEFT + B
        elif left:
            return 6  # LEFT
        elif a_button:
            return 5  # A (jump)
        elif down:
            return 10  # DOWN
        elif up:
            return 11  # UP
        else:
            return 0  # NOOP
    
    def setup_quit_handler(self):
        """Setup ESC key handler to quit game."""
        keyboard.add_hotkey('esc', self.quit_handler)
    
    def quit_handler(self):
        """Handle quit command."""
        self.quit_game = True
        print("\nQuitting game...")

class SaveStateManager:
    """Manage saving and organizing RAM states."""
    
    def __init__(self, base_dir="save_states"):
        self.base_dir = base_dir
        self.ensure_base_dir()
        
    def ensure_base_dir(self):
        """Create base directory if it doesn't exist."""
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            print(f"Created save states directory: {self.base_dir}")
    
    def get_stage_info(self, info):
        """Extract stage information from environment info."""
        world = info.get('world', 1)
        stage = info.get('stage', 1)
        return world, stage
    
    def get_stage_dir(self, world, stage):
        """Get directory path for a specific stage."""
        stage_dir = os.path.join(self.base_dir, f"world_{world}_stage_{stage}")
        if not os.path.exists(stage_dir):
            os.makedirs(stage_dir)
        return stage_dir
    
    def get_next_save_number(self, stage_dir):
        """Get the next available save number for a stage."""
        existing_saves = [f for f in os.listdir(stage_dir) if f.startswith('ram_') and f.endswith('.npy')]
        if not existing_saves:
            return 1
        
        numbers = []
        for save_file in existing_saves:
            try:
                num = int(save_file.split('_')[1].split('.')[0])
                numbers.append(num)
            except (ValueError, IndexError):
                continue
        
        return max(numbers) + 1 if numbers else 1
    
    def save_state(self, env, info, steps, score):
        """Save current environment RAM state."""
        try:
            # Get stage information
            world, stage = self.get_stage_info(info)
            stage_dir = self.get_stage_dir(world, stage)
            save_num = self.get_next_save_number(stage_dir)
            
            # Find the underlying NES environment to get RAM
            nes_env = env
            while hasattr(nes_env, 'env'):
                nes_env = nes_env.env
            
            # Get the RAM array (2KB for NES)
            ram_array = None
            if hasattr(nes_env, 'ram'):
                ram_array = nes_env.ram.copy()
            elif hasattr(nes_env, '_ram'):
                ram_array = nes_env._ram.copy()
            elif hasattr(nes_env, 'unwrapped') and hasattr(nes_env.unwrapped, 'ram'):
                ram_array = nes_env.unwrapped.ram.copy()
            else:
                print("❌ Could not access RAM array")
                return None
            
            # Convert to numpy array if it isn't already
            if not isinstance(ram_array, np.ndarray):
                ram_array = np.array(ram_array)
            
            # Save RAM as numpy array
            ram_filename = f"ram_{save_num:03d}.npy"
            ram_path = os.path.join(stage_dir, ram_filename)
            np.save(ram_path, ram_array)
            
            print(f"💾 SAVED RAM STATE: World {world}-{stage}, Save #{save_num}")
            print(f"   Location: X={info.get('x_pos', 0)}, Score={score}, Lives={info.get('life', 0)}")
            print(f"   RAM size: {ram_array.shape}, File: {ram_path}")
            
            return ram_path
            
        except Exception as e:
            print(f"❌ Error saving RAM state: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def list_saves(self):
        """List all available RAM saves organized by stage."""
        if not os.path.exists(self.base_dir):
            print("No RAM save states found.")
            return
        
        print("\n📁 AVAILABLE RAM SAVE STATES:")
        print("=" * 60)
        
        total_saves = 0
        for stage_dir in sorted(os.listdir(self.base_dir)):
            stage_path = os.path.join(self.base_dir, stage_dir)
            if os.path.isdir(stage_path):
                saves = [f for f in os.listdir(stage_path) if f.startswith('ram_') and f.endswith('.npy')]
                if saves:
                    print(f"\n{stage_dir.upper().replace('_', ' ')}:")
                    for save_file in sorted(saves):
                        print(f"  • {save_file}")
                        total_saves += 1
        
        if total_saves == 0:
            print("No RAM save states found yet. Press F2 during gameplay to save!")
        else:
            print(f"\nTotal: {total_saves} RAM save states")
        print("=" * 60)
    
    def load_ram_state(self, world, stage, save_num):
        """Load a specific RAM state for use in RL training."""
        try:
            stage_dir = self.get_stage_dir(world, stage)
            ram_filename = f"ram_{save_num:03d}.npy"
            ram_path = os.path.join(stage_dir, ram_filename)
            
            if not os.path.exists(ram_path):
                print(f"❌ RAM file not found: {ram_path}")
                return None
            
            # Load RAM array
            ram_array = np.load(ram_path)
            
            return ram_array
            
        except Exception as e:
            print(f"❌ Error loading RAM state: {e}")
            return None
    
    def get_all_saves_for_stage(self, world, stage):
        """Get all available save files for a specific stage."""
        stage_dir = self.get_stage_dir(world, stage)
        if not os.path.exists(stage_dir):
            return []
        
        saves = []
        ram_files = [f for f in os.listdir(stage_dir) if f.startswith('ram_') and f.endswith('.npy')]
        
        for ram_file in sorted(ram_files):
            save_num = int(ram_file.split('_')[1].split('.')[0])
            
            saves.append({
                'save_num': save_num,
                'ram_file': os.path.join(stage_dir, ram_file)
            })
        
        return saves

def print_controls():
    """Print the control scheme."""
    print("\n" + "="*60)
    print("SUPER MARIO BROS - KEYBOARD CONTROLS")
    print("="*60)
    print("Movement:")
    print("  ← → ↑ ↓     : Arrow Keys")
    print("Actions:")
    print("  SPACE       : A Button (Jump)")
    print("  SHIFT       : B Button (Run/Fireball)")
    print("Combinations:")
    print("  →  + SPACE  : Run and Jump")
    print("  →  + SHIFT  : Run Right")
    print("  SHIFT+SPACE : Run and Jump")
    print("Save States:")
    print("  F2          : Save current RAM state")
    print("  F5          : Reset environment")
    print("Other:")
    print("  ESC         : Quit Game")
    print("="*60)
    print("TIP: Hold SHIFT to run faster and jump further!")
    print("💾 Save states are organized by World/Stage automatically")
    print("="*60 + "\n")

def main():
    """Main game loop."""
    print("Initializing Super Mario Bros Human Player...")
    print("Note: You may need to run this script as administrator for keyboard access.")
    
    # Create keyboard controller and save state manager
    controller = KeyboardController()
    controller.setup_quit_handler()
    save_manager = SaveStateManager()
    
    # Create environment
    try:
        env = create_human_env()
        print("Environment created successfully!")
    except Exception as e:
        print(f"Error creating environment: {e}")
        return
    
    print_controls()
    
    # Show existing saves
    save_manager.list_saves()
    
    # Game statistics
    total_episodes = 0
    best_score = 0
    best_distance = 0
    
    try:
        while not controller.quit_game:
            # Reset environment for new episode
            obs = env.reset()
            done = False
            total_reward = 0
            steps = 0
            episode_start_time = time.time()
            
            total_episodes += 1
            print(f"\nStarting Episode {total_episodes}")
            print("Game starting in 3 seconds... Press ESC to quit anytime.")
            
            # Brief countdown
            for i in range(3, 0, -1):
                if controller.quit_game:
                    break
                print(f"{i}...")
                time.sleep(1)
            
            if controller.quit_game:
                break
                
            print("GO!")
            
            while not done and not controller.quit_game:
                # Get keyboard action
                action = controller.get_action()
                
                # Check for quit
                if action == -1 or controller.quit_game:
                    print("\nQuitting game...")
                    done = True
                    break
                
                # Check for reset
                if action == -2:
                    print("\n🔄 RESETTING ENVIRONMENT...")
                    obs = env.reset()
                    steps = 0
                    total_reward = 0
                    episode_start_time = time.time()
                    print("Environment reset! Starting fresh.")
                    continue
                
                # Check for save state
                if action == -3:
                    print("\n💾 SAVING RAM STATE...")
                    save_manager.save_state(env, info if 'info' in locals() else {}, steps, info.get('score', 0) if 'info' in locals() else 0)
                    continue
                
                # Take action in environment
                try:
                    obs, reward, done, info = env.step(action)
                    
                    # Render the environment
                    env.render()
                    
                    # Update statistics
                    total_reward += reward
                    steps += 1
                    
                    # Print game info periodically
                    if steps % 100 == 0:
                        print(f"Steps: {steps}, Score: {info.get('score', 0)}, "
                              f"Lives: {info.get('life', 0)}, X-pos: {info.get('x_pos', 0)}")
                    
                    # Check for level completion
                    if info.get('flag_get', False):
                        print(f"\n🎉 LEVEL COMPLETED! 🎉")
                        print(f"Final Score: {info.get('score', 0)}")
                        print(f"Steps taken: {steps}")
                        time.sleep(2)  # Pause to show completion
                    
                    # Small delay to control game speed
                    time.sleep(0.016)  # ~60 FPS
                    
                except Exception as e:
                    print(f"Error during game step: {e}")
                    break
            
            if controller.quit_game:
                break
                
            # Episode ended - show statistics
            episode_time = time.time() - episode_start_time
            final_score = info.get('score', 0) if 'info' in locals() else 0
            final_distance = info.get('x_pos', 0) if 'info' in locals() else 0
            
            # Update best scores
            if final_score > best_score:
                best_score = final_score
            if final_distance > best_distance:
                best_distance = final_distance
            
            print(f"\n" + "="*40)
            print(f"EPISODE {total_episodes} COMPLETE")
            print(f"="*40)
            print(f"Time: {episode_time:.1f} seconds")
            print(f"Steps: {steps}")
            print(f"Final Score: {final_score}")
            print(f"Distance Reached: {final_distance}")
            print(f"Best Score: {best_score}")
            print(f"Best Distance: {best_distance}")
            print(f"="*40)
            
            # Ask if player wants to continue
            print("\nPress 'R' to restart, ESC to quit, or ENTER to continue...")
            
            # Wait for input
            waiting = True
            while waiting and not controller.quit_game:
                if keyboard.is_pressed('r'):
                    print("Restarting...")
                    waiting = False
                    time.sleep(0.5)  # Prevent multiple triggers
                elif keyboard.is_pressed('enter'):
                    print("Continuing...")
                    waiting = False
                    time.sleep(0.5)  # Prevent multiple triggers
                elif controller.quit_game:
                    waiting = False
                
                time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nGame interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up
        try:
            env.close()
        except:
            pass
        print("Game closed. Thanks for playing!")

if __name__ == "__main__":
    main() 