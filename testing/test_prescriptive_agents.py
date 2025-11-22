"""
Test Prescriptive Agent
"""

from src.agents import PrescriptiveAgent
import numpy as np

print("=" * 60)
print("TESTING PRESCRIPTIVE AGENT (RL)")
print("=" * 60)

# Initialize agent
agent = PrescriptiveAgent()

# Simulate 100 iterations
print("\n🎯 Running 100 iterations of RL learning...")

for i in range(10000):
    # Create sample player data
    player_data = {
        'age': np.random.randint(18, 60),
        'playtime_hours': np.random.uniform(10, 500),
        'sessions_per_week': np.random.randint(1, 20),
        'player_level': np.random.randint(1, 100),
        'has_purchases': np.random.random() > 0.5,
        'predicted_engagement': np.random.randint(0, 3)
    }
    
    # Get recommendation
    result = agent.execute({
        'mode': 'recommend',
        'player_data': player_data
    })
    
    action_id = result['data']['action_id']
    
    # Simulate reward (in reality, this comes from ExecutionAgent)
    # Higher rewards for appropriate actions
    if player_data['predicted_engagement'] == 0:  # Low engagement
        reward = 2 if action_id in [0, 1, 2] else -1  # Discounts/notifications good
    elif player_data['predicted_engagement'] == 1:  # Medium
        reward = 2 if action_id in [2, 3] else 0  # Notifications/content good
    else:  # High engagement
        reward = 1 if action_id == 4 else -1  # No action best
    
    # Update bandit
    agent.execute({
        'mode': 'update',
        'action_id': action_id,
        'reward': reward
    })
    
    if (i + 1) % 20 == 0:
        print(f"  Iteration {i+1}/100 complete...")

# Get final statistics
print("\n📊 FINAL RL STATISTICS:")
stats_result = agent.execute({'mode': 'stats'})

print(f"\n  🏆 Best Action: {stats_result['data']['best_action_name']}")
print(f"  🔄 Total Iterations: {stats_result['data']['total_iterations']}")

print("\n  📈 Action Performance:")
for action_id, stats in stats_result['data']['action_stats'].items():
    action_name = agent.actions[action_id]['name']
    print(f"    {action_name}:")
    print(f"      Count: {stats['count']}")
    print(f"      Avg Reward: {stats['avg_reward']:.3f}")
    print(f"      Theta: {stats['theta']:.3f}")

print("\n" + "=" * 60)
print("✅ PRESCRIPTIVE AGENT TEST COMPLETE!")
print("=" * 60)