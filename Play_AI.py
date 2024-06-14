import pygame
from pong import Game
from main import PongGame
import pickle
import neat
import os

def test_ai(config):
    width, height = 700, 500
    window = pygame.display.set_mode((width, height))
    with open("best.pickle2","rb") as f:
        winner =pickle.load(f)
    game =PongGame(window,width,height)
    game.test_ai(winner,config)

if __name__ == "__main__":
    loca_dir = os.path.dirname(__file__)
    config_path = os.path.join(loca_dir, "../NEAT-Ping_Pong/config.txt")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    test_ai(config)
