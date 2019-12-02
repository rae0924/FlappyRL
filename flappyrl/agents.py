import neat, cv2, os, pickle, time
import numpy as np
from ple.games.flappybird import FlappyBird
from ple import PLE


class FlappyBirdAgent(object):
    def __init__(self, display_screen=True):
        self.display_screen = display_screen
        self.game = FlappyBird()
        self.env = PLE(game=self.game, fps=30, display_screen=self.display_screen)
        self.action_space = self.env.getActionSet()
        self.observation_space_shape = self.env.getGameState().__len__()

    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            self.env.init()
            net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
            done = False
            fitness = 0
            while not done:
                if self.display_screen:
                    time.sleep(1.0/100)
                observation = self.env.getGameState()
                observation = np.array(list(observation.values()))
                action = net.activate(observation)
                action = np.argmax(action)
                action = self.action_space[action]
                reward = self.env.act(action)
                if reward > 0.0:
                    fitness += reward * 10
                fitness += 1.0
                if self.env.game_over():
                    done = True
            print(genome_id, fitness)
            genome.fitness = fitness

    def run(self, config_file):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        winner = p.run(self.eval_genomes)
        with open("flappy_bird_agent.pkl", "wb") as f:
            pickle.dump(winner, f)

class FlappyBirdAgentVis(FlappyBirdAgent):
    def __init__(self, display_screen=True):
        super().__init__(display_screen)
        self.observation_space_shape = cv2.transpose(self.env.getScreenGrayscale()).shape
        # 36 by 16 image
        self.inx = int(self.observation_space_shape[1] / 8)
        self.iny = int(self.observation_space_shape[0] / 32)

    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            self.env.init()
            net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
            cv2.namedWindow('main', cv2.WINDOW_NORMAL)
            done = False
            fitness = 0
            while not done:
                observation = self.env.getScreenGrayscale()
                observation = cv2.transpose(observation)
                observation = cv2.resize(observation, (self.inx, self.iny))
                cv2.imshow('main', observation)
                imgarray = np.ndarray.flatten(observation)
                action = net.activate(imgarray)
                action = np.argmax(action)
                action = self.action_space[action]
                reward = self.env.act(action)
                if reward > 0.0:
                    fitness += reward * 10
                fitness += 1.0
                if self.env.game_over():
                    done = True
            print(genome_id, fitness)
            genome.fitness = fitness
        
    def run(self, config_file, gen=100):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                    neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        winner = p.run(self.eval_genomes, n=gen)
        with open("flappy_bird_agent_vis.pkl", "wb") as f:
            pickle.dump(winner, f)
