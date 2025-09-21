import os
from Democracy_env import DemocracyEnv
from stable_baselines3 import PPO
import time

def run_model(model_path, num_episodes=1):
    # 1. Create the environment
    env = DemocracyEnv()
    
    # 2. Load the model from ZIP file
    try:
        print(f" Loading model from {model_path}...")
        model = PPO.load(model_path, env=env)
        print(" Model loaded successfully")
    except Exception as e:
        print(f" Error loading model: {str(e)}")
        return

    # 3. Run the model for each episode
    for episode in range(num_episodes):
        print(f"\n=== Episode {episode+1}/{num_episodes} ===")
        
        obs, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0
        
        start_time = time.time()
        
        while not (terminated or truncated):
            # Predict action using the model
            action, _states = model.predict(obs, deterministic=False)
            
            # Take the action in the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Update statistics
            total_reward += reward
            step_count += 1
            
            # Display step information
            print(f"Step {step_count} | Reward: {reward:.2f} | Total: {total_reward:.2f}")
            
            # Delay for better step observation
            time.sleep(0.5)
        
        # Calculate runtime
        duration = time.time() - start_time
        
        # Display episode results
        print(f"\n⏱️ Runtime: {duration:.1f} seconds")
        print(f" Steps taken: {step_count}")
        print(f" Total reward: {total_reward:.2f}")
        print("=" * 40)

if __name__ == "__main__":
    # Set the model path
    model_path = "ppo_democracy4_final.zip"
    
    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f" Model file not found: {model_path}")
        exit(1)
    
    # Run the model for 3 episodes
    run_model(model_path, num_episodes=3)
