import neat, time
import numpy as np
from ple import PLE 
from ple.games.flappybird import FlappyBird

class Evaluator(object):
    def __init__(self, config_file):
        self.game = FlappyBird()
        self.env = PLE(game=self.game, fps=30, display_screen=True)
        self.observation_space_shape = self.env.getGameState().__len__()
        self.action_space = self.env.getActionSet()
    
    def evaluate_agent(self, genome, config_file):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
        self.net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        done = False
        fitness = 0
        score = 0
        while not done:
            time.sleep(1.0/100)
            observation = self.env.getGameState()
            observation = np.array(list(observation.values()))
            action = self.net.activate(observation)
            action = np.argmax(action)
            action = self.action_space[action]
            reward = self.env.act(action)
            if reward > 0.0:
                fitness += reward * 10
                score += reward
            fitness += 1.0
            if self.env.game_over():
                done = True
        print(f"Fitness: {fitness}")
        print(f"Score: {int(score)}")

    def run_agent(self, genome, config_file):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
        self.net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        done = False
        while not done:
            time.sleep(1.0/100)
            observation = self.env.getGameState()
            observation = np.array(list(observation.values()))
            action = self.net.activate(observation)
            action = np.argmax(action)
            action = self.action_space[action]
            __ = self.env.act(action)
            if self.env.game_over():
                done = True