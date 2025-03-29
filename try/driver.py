import numpy as np
import matplotlib.pyplot as plt
from data_collection0 import Env

def test_environment():
    """
    Test function for the Env class from data_collection0.py
    """
    # Create the environment
    env = Env(log_dir='./test_logs')
    
    # Reset the environment to initialize state
    state = env.reset()
    
    print(f"Environment initialized with {env.num_uavs} UAVs")
    print(f"Map dimensions: {env.mapx}x{env.mapy}")
    print(f"Initial UAV positions: {env.uav}")
    print(f"State shape: {[s.shape for s in state]}")
    
    # Run a few random steps
    total_reward = 0
    num_steps = 20
    
    # Store trajectories for visualization
    trajectories = [[] for _ in range(env.num_uavs)]
    for i in range(env.num_uavs):
        trajectories[i].append(env.uav[i].copy())
    
    # Run simulation with random actions
    for step in range(num_steps):
        # Generate random actions for each UAV
        actions = []
        for i in range(env.num_uavs):
            # Random direction with random magnitude within maxdistance
            action = np.random.uniform(-1, 1, size=(2,))
            # Normalize and scale
            norm = np.linalg.norm(action)
            if norm > 0:
                action = action / norm * np.random.uniform(0, env.maxdistance)
            actions.append(action)
        
        # Step the environment
        next_state, rewards, done, info, _ = env.step(actions)
        
        # Store trajectory information
        for i in range(env.num_uavs):
            trajectories[i].append(env.uav[i].copy())
        
        # Print step information
        step_reward = sum(rewards)
        total_reward += step_reward
        # print(f"Step {step+1}: Reward = {step_reward:.4f}, Done = {done}")
        # print(f"  UAV positions: {env.uav}")
        # print(f"  UAV energy levels: {[e/env.maxenergy for e in env.energy]}")
        
        # Break if done
        if done:
            print("Environment signaled done")
            break
    
    # Print final statistics
    print("\nSimulation completed")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Data collected: {env.collection}")
    print(f"Collection fairness: {env.collection_fairness:.4f}")
    print(f"Remaining rewards: {env.leftrewards:.4f}")
    
    # Plot trajectories
    plt.figure(figsize=(10, 10))
    
    # Plot data points
    for i, data_point in enumerate(env.datas):
        plt.scatter(data_point[1], data_point[0], 
                   s=30*env._mapmatrix[i], 
                   alpha=0.5, 
                   color='blue')
    
    # Plot UAV trajectories
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    for i, traj in enumerate(trajectories):
        traj_array = np.array(traj)
        plt.plot(traj_array[:, 1], traj_array[:, 0], 
                 marker='o', markersize=5, 
                 color=colors[i % len(colors)], 
                 label=f'UAV {i+1}')
        
    plt.xlim(0, env.mapy)
    plt.ylim(0, env.mapx)
    plt.grid(True)
    plt.legend()
    plt.title('UAV Trajectories')
    plt.xlabel('Y coordinate')
    plt.ylabel('X coordinate')
    plt.gca().invert_yaxis()  # Invert y-axis to match grid coordinates
    
    plt.savefig('uav_trajectory.png')
    plt.show()

if __name__ == "__main__":
    test_environment()