import flappyrl
import os, pickle

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), 'config-flappy-bird-agent')
    # agent = flappyrl.agents.FlappyBirdAgent(display_screen=False)
    # agent.run(config_file=config_path)
    with open('flappy_bird_agent.pkl', 'rb') as f:
        winner = pickle.load(f)
    evaluator = flappyrl.evaluate.Evaluator(config_path)
    evaluator.evaluate_agent(winner, config_path)