#!/usr/bin/env python3
"""
Mario Bros Human Player Script
Play Super Mario Bros using keyboard controls.
Uses the same environment as the RL training but simplified for human play.
Now includes comprehensive gameplay recording for random start location generation.
"""

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import pygame
import time
import os
import json
from datetime import datetime
from config import ONE_RECORDING_PER_STAGE

def create_human_env(world=1, stage=1):
    """Create a Mario environment optimized for human play."""
    # Create environment for specific world-stage
    env_name = f'SuperMarioBros-{world}-{stage}-v0'
    try:
        env = gym_super_mario_bros.make(env_name)
    except:
        print(f"Warning: Level {world}-{stage} not found, defaulting to 1-1")
        env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    
    # Add SkipFrame wrapper to match training environment
    # This ensures recorded actions will work correctly when replayed
    from environment import SkipFrame
    env = SkipFrame(env, skip=6)  # Same skip value as training
    
    return env

class ActionRecorder:
    """Record sequences of actions for generating random start positions."""
    
    def __init__(self, base_dir="recorded_gameplay"):
        self.base_dir = base_dir
        self.ensure_base_dir()
        self.recording = False
        self.current_recording = None
        self.reset_recording()
        
    def ensure_base_dir(self):
        """Create base directory if it doesn't exist."""
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            print(f"Created recorded gameplay directory: {self.base_dir}")
    
    def reset_recording(self):
        """Reset the current recording session."""
        self.current_recording = {
            'actions': [],
            'world': 1,
            'stage': 1,
            'start_time': datetime.now().isoformat(),
            'session_id': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
    def start_recording(self, world, stage):
        """Start recording actions."""
        self.recording = True
        self.current_recording['world'] = world
        self.current_recording['stage'] = stage
        # Note: Less prominent message since recording is now automatic
        
    def stop_recording(self):
        """Stop recording actions."""
        self.recording = False
        print("🛑 STOPPED RECORDING")
        
    def record_action(self, action):
        """Record a single action."""
        if not self.recording:
            return
        self.current_recording['actions'].append(action)
    
    def save_recording(self, final_info=None):
        """Save the recorded action sequence to disk."""
        if not self.current_recording['actions']:
            print("⚠️ No actions recorded, skipping save")
            return None
            
        # Get session info
        world = self.current_recording['world']
        stage = self.current_recording['stage']
        session_id = self.current_recording['session_id']
        
        # Create directory for this world/stage
        session_dir = os.path.join(self.base_dir, f"world_{world}_stage_{stage}")
        if not os.path.exists(session_dir):
            os.makedirs(session_dir)
            
        # If ONE_RECORDING_PER_STAGE is True, delete existing recordings
        if ONE_RECORDING_PER_STAGE:
            existing_files = [f for f in os.listdir(session_dir) if f.startswith('actions_') and f.endswith('.json')]
            for existing_file in existing_files:
                old_path = os.path.join(session_dir, existing_file)
                try:
                    os.remove(old_path)
                    print(f"🗑️ Removed old recording: {existing_file}")
                except Exception as e:
                    print(f"⚠️ Failed to remove old recording {existing_file}: {e}")
            
        # Save actions as simple list
        actions_filename = f"actions_{session_id}.json"
        actions_path = os.path.join(session_dir, actions_filename)
        
        recording_data = {
            'actions': self.current_recording['actions'],
            'world': world,
            'stage': stage,
            'start_time': self.current_recording['start_time'],
            'session_id': session_id,
            'total_actions': len(self.current_recording['actions']),
            'final_info': final_info if final_info else {}
        }
        
        with open(actions_path, 'w') as f:
            json.dump(recording_data, f, indent=2)
            
        print(f"💾 SAVED ACTION RECORDING - {len(self.current_recording['actions'])} actions")
        print(f"   File: {actions_path}")
        
        return actions_path
    
    def list_recorded_actions(self):
        """List all recorded action sequences."""
        if not os.path.exists(self.base_dir):
            print("No recorded actions found.")
            return
            
        print("\n🎬 RECORDED ACTION SEQUENCES:")
        print("=" * 70)
        
        total_recordings = 0
        for stage_dir in sorted(os.listdir(self.base_dir)):
            stage_path = os.path.join(self.base_dir, stage_dir)
            if os.path.isdir(stage_path):
                action_files = [f for f in os.listdir(stage_path) if f.startswith('actions_') and f.endswith('.json')]
                if action_files:
                    print(f"\n{stage_dir.upper().replace('_', ' ')}:")
                    for action_file in sorted(action_files):
                        try:
                            with open(os.path.join(stage_path, action_file), 'r') as f:
                                data = json.load(f)
                                final_score = data.get('final_info', {}).get('score', 'N/A')
                                final_x_pos = data.get('final_info', {}).get('x_pos', 'N/A')
                                print(f"  • {data['session_id']}: {data['total_actions']} actions, "
                                      f"final score: {final_score}, x_pos: {final_x_pos}")
                                total_recordings += 1
                        except Exception as e:
                            print(f"  • {action_file}: Error reading file")
        
        if total_recordings == 0:
            print("No recorded actions found yet. Play and complete a level!")
        else:
            print(f"\nTotal: {total_recordings} recorded action sequences")
        print("=" * 70)

class KeyboardController:
    """Handle keyboard input for Mario controls using pygame."""
    
    def __init__(self):
        self.current_action = 0
        self.quit_game = False
        self.reset_requested = False
        self.last_reset_time = 0
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
        
        # Initialize pygame for keyboard input
        pygame.init()
        # Create a small dummy display for pygame to work
        self.screen = pygame.display.set_mode((1, 1))
        pygame.display.set_caption("Mario Controller")
        
    def get_action(self):
        """Get current action based on pressed keys."""
        if self.quit_game:
            return -1
        
        # Process pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_game = True
                return -1
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.quit_game = True
                    print("\nQuitting game...")
                    return -1
                elif event.key == pygame.K_F5 or event.key == pygame.K_r:
                    # Debounce reset key
                    current_time = time.time()
                    if current_time - self.last_reset_time > 0.5:
                        self.last_reset_time = current_time
                        return -2  # Special reset action
        
        # Get current key states
        keys = pygame.key.get_pressed()
        
        # Check key combinations in priority order
        right = keys[pygame.K_RIGHT]
        left = keys[pygame.K_LEFT]
        up = keys[pygame.K_UP]
        down = keys[pygame.K_DOWN]
        a_button = keys[pygame.K_SPACE]  # Jump
        b_button = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]  # Run/Fire
        
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
        """Setup quit handling - now handled in get_action()."""
        pass  # No separate setup needed with pygame
    
    def cleanup(self):
        """Clean up pygame resources."""
        pygame.quit()



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
    print("Controls:")
    print("  F5 or R     : Reset environment")
    print("  ESC         : Quit Game")
    print("="*60)
    print("TIP: Hold SHIFT to run faster and jump further!")
    print("🎬 Actions are automatically recorded from start to flag")
    print("🏁 When you reach the flag, you'll be asked to save the recording")
    print("📺 A small pygame window will appear - keep it focused for input")
    print("="*60 + "\n")

def select_level():
    """Let user select which level to play."""
    print("\n" + "="*50)
    print("LEVEL SELECTION")
    print("="*50)
    print("Available levels: World 1-8, Stage 1-4")
    print("Examples: 1-1, 2-3, 8-4")
    print("="*50)
    
    while True:
        try:
            level_input = input("Enter level (e.g., 1-1) or press Enter for 1-1: ").strip()
            if not level_input:
                return 1, 1  # Default to 1-1
            
            if '-' not in level_input:
                print("Invalid format. Use format like '1-1'")
                continue
                
            parts = level_input.split('-')
            if len(parts) != 2:
                print("Invalid format. Use format like '1-1'")
                continue
                
            world = int(parts[0])
            stage = int(parts[1])
            
            if world < 1 or world > 8 or stage < 1 or stage > 4:
                print("Invalid level. World must be 1-8, Stage must be 1-4")
                continue
                
            return world, stage
            
        except ValueError:
            print("Invalid input. Use numbers like '1-1'")
        except (EOFError, KeyboardInterrupt):
            print("\nDefaulting to level 1-1")
            return 1, 1

def main():
    """Main game loop."""
    print("Initializing Super Mario Bros Human Player...")
    print("Using pygame for cross-platform keyboard input.")
    
    # Select level to play
    selected_world, selected_stage = select_level()
    print(f"\n🎮 Selected Level: World {selected_world}-{selected_stage}")
    
    # Create keyboard controller and action recorder
    controller = KeyboardController()
    controller.setup_quit_handler()
    action_recorder = ActionRecorder()
    
    # Create environment for selected level
    try:
        env = create_human_env(selected_world, selected_stage)
        print("Environment created successfully!")
    except Exception as e:
        print(f"Error creating environment: {e}")
        return
    
    print_controls()
    
    # Show existing recordings
    action_recorder.list_recorded_actions()
    
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
            
            # Reset recorder and start recording automatically
            action_recorder.reset_recording()
            # Use selected level for recording
            action_recorder.start_recording(selected_world, selected_stage)
            print(f"🎬 Recording World {selected_world}-{selected_stage}")
            
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
                    # Stop current recording
                    if action_recorder.recording:
                        action_recorder.stop_recording()
                    
                    obs = env.reset()
                    steps = 0
                    total_reward = 0
                    episode_start_time = time.time()
                    
                    # Clear and restart recording for selected level
                    action_recorder.reset_recording()
                    action_recorder.start_recording(selected_world, selected_stage)
                    print(f"🎬 Recording World {selected_world}-{selected_stage} (RESET)")
                    
                    print("Environment reset! Starting fresh.")
                    continue
                

                
                # Take action in environment
                try:
                    # Record action if recording is active
                    if action_recorder.recording:
                        action_recorder.record_action(action)
                    
                    obs, reward, done, info = env.step(action)
                    
                    # Render the environment
                    env.render()
                    
                    # Update statistics
                    total_reward += reward
                    steps += 1
                    
                    # Print game info periodically
                    if steps % 100 == 0:
                        print(f"Steps: {steps}, Score: {info.get('score', 0)}, "
                              f"Lives: {info.get('life', 0)}, X-pos: {info.get('x_pos', 0)} 🎬")
                    
                    # Check for level completion (improved flag detection)
                    x_pos = info.get('x_pos', 0)
                    flag_reached = (x_pos >= 3160 or info.get('flag_get', False))
                    
                    if flag_reached:
                        print(f"\n🎉 LEVEL COMPLETED! 🎉")
                        print(f"Final Score: {info.get('score', 0)}")
                        print(f"Steps taken: {steps}")
                        print(f"Final x-position: {x_pos}")
                        
                        # Close pygame display immediately
                        try:
                            pygame.display.quit()
                            pygame.quit()
                            print("🖼️ Display closed.")
                        except:
                            pass
                        
                        # Stop recording and ask if user wants to save
                        if action_recorder.recording:
                            action_recorder.stop_recording()
                            
                            # Switch to console mode for user interaction
                            print("\n" + "="*50)
                            print("🎬 RECORDING COMPLETED!")
                            print(f"Actions recorded: {len(action_recorder.current_recording['actions'])}")
                            print(f"Final position: {info.get('x_pos', 0)}")
                            print(f"Final score: {info.get('score', 0)}")
                            print("="*50)
                            
                            while True:
                                try:
                                    save_choice = input("Save this recording? (y/n): ").lower().strip()
                                    if save_choice in ['y', 'yes']:
                                        action_recorder.save_recording(info)
                                        print("✅ Recording saved!")
                                        break
                                    elif save_choice in ['n', 'no']:
                                        print("❌ Recording discarded.")
                                        break
                                    else:
                                        print("Please enter 'y' or 'n'")
                                except (EOFError, KeyboardInterrupt):
                                    print("\n❌ Recording discarded.")
                                    break
                        
                        # End the game completely after flag completion
                        controller.quit_game = True
                        done = True
                    
                    # Delay to match original NES game speed
                    # Since we use SkipFrame(skip=6), we need 6x longer delay
                    # Original NES: 60 FPS = 0.0167s per frame
                    # With skip=6: 6 * 0.0167 = 0.1s per action
                    time.sleep(0.1)  # ~10 actions/second (matching original game feel)
                    
                except Exception as e:
                    print(f"Error during game step: {e}")
                    break
            
            if controller.quit_game:
                break
                
            # Stop recording if still active (without saving - only flag completion saves)
            if action_recorder.recording:
                action_recorder.stop_recording()
                print("🎬 Recording stopped (not saved - only flag completion saves recordings)")
                
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
            last_input_time = 0
            while waiting and not controller.quit_game:
                # Process pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        controller.quit_game = True
                        waiting = False
                    elif event.type == pygame.KEYDOWN:
                        current_time = time.time()
                        if current_time - last_input_time > 0.5:  # Debounce
                            if event.key == pygame.K_r:
                                print("Restarting...")
                                waiting = False
                                last_input_time = current_time
                            elif event.key == pygame.K_RETURN:
                                print("Continuing...")
                                waiting = False
                                last_input_time = current_time
                            elif event.key == pygame.K_ESCAPE:
                                controller.quit_game = True
                                waiting = False
                
                time.sleep(0.1)
            
            # Don't recreate environment after flag completion - game ends
    
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
        
        try:
            controller.cleanup()
        except:
            pass
        
        print("Game closed. Thanks for playing!")

if __name__ == "__main__":
    main() 