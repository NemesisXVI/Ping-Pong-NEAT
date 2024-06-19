import pygame
from pong import Game
import neat
import os
import pickle

width, height = 700, 500
window = pygame.display.set_mode((width, height))

game = Game(window, width, height)


class PongGame:
    def __init__(self, window, width, height):
        self.game = Game(window, width, height)
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle
        self.ball = self.game.ball

    # Function to input 2 genomes and let them play the game.
    def train_ai(self, genome1, genome2, config):
        # Creating neural networks for each genome
        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        net2 = neat.nn.FeedForwardNetwork.create(genome2, config)

        run = True
        # Letting the 2 genomes play the game until one of them looses.
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()

            # Giving the 3 inputs to the network :
            # 1. y coordinate of paddle  2. y coordinate of ball  3. absolute difference between the x coordinates of the paddle and the ball.
            output1 = net1.activate((self.left_paddle.y, self.ball.y, abs(self.left_paddle.x - self.ball.x)))

            # The neural network will output 3 values:
            # 1. Probability to not move   2. Probability to move up   3. Probability to move down.
            # Moving the paddles accordingly.
            decision1 = output1.index(max(output1))
            if decision1 == 0:
                pass
            elif decision1 == 1:
                self.game.move_paddle(left=True, up=True)
            else:
                self.game.move_paddle(left=True, up=False)

            # Similar to the left paddle
            output2 = net2.activate((self.right_paddle.y, self.ball.y, abs(self.right_paddle.x - self.ball.x)))
            decision2 = output2.index(max(output2))
            if decision2 == 0:
                pass
            elif decision2 == 1:
                self.game.move_paddle(left=False, up=True)
            else:
                self.game.move_paddle(left=False, up=False)

            # Executing a single game loop.
            game_info = self.game.loop()

            self.game.draw(draw_score=False, draw_hits=True)
            pygame.display.update()

            # When either AI fails to hit the ball, we calculate fitness and break this game loop. Or when more than 50 hits have been made.
            if game_info.left_score >= 1 or game_info.right_score >= 1 or game_info.left_hits > 50:
                self.calculate_fitness(genome1, genome2, game_info)
                break

    # Function to calculate the fitness of a genome, which is determined by the number of times it hits the ball.
    def calculate_fitness(self, genome1, genome2, game_info):
        genome1.fitness += game_info.left_hits
        genome2.fitness += game_info.right_hits

    # Used in Play_AI
    # Functionality same as that of train_ai. Only the right paddle controlled by AI, left is controlled by user.
    def test_ai(self, genome, config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        run = True
        clock = pygame.time.Clock()
        while run:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                self.game.move_paddle(left=True, up=True)
            if keys[pygame.K_s]:
                self.game.move_paddle(left=True, up=False)

            output = net.activate((self.right_paddle.y, self.ball.y, abs(self.right_paddle.x - self.ball.x)))
            decision = output.index(max(output))
            if decision == 0:
                pass
            elif decision == 1:
                self.game.move_paddle(left=False, up=True)
            else:
                self.game.move_paddle(left=False, up=False)

            game_info = self.game.loop()

            self.game.draw(True, False)
            pygame.display.update()

        pygame.quit()


# A function to let each genome of the population play the game and get a corresponding fitness.
# We let each genome play against each other hence the nested loop.
def eval_genomes(genomes, config):
    width, height = 700, 500
    window = pygame.display.set_mode((width, height))

    for i, (genome_id1, genome1) in enumerate(genomes):
        if i == len(genomes) - 1:
            break
        genome1.fitness = 0
        for genome_id2, genome2 in genomes[i + 1:]:
            if genome2.fitness == None:
                genome2.fitness = 0
            game = PongGame(window, width, height)
            game.train_ai(genome1, genome2, config)


# Function to run the neat algorithm enabled with Statistics reporter to give summary of stats for each generation.
def run_neat(config):
    p = neat.Population(config)
    # p=neat.Checkpointer.restore_checkpoint('neat-checkpoint-35')
    p.add_reporter((neat.StdOutReporter(True)))
    stats = neat.StatisticsReporter()
    p.add_reporter((stats))
    p.add_reporter(neat.Checkpointer(1))

    # Running the neat algorithm for 50 generation or until the fitness threshold is met. The neural network with the best performance is stored as winner.
    winner = p.run(eval_genomes, 50)
    # Storing the best AI for using later.
    with open("best.pickle2", "wb") as f:
        pickle.dump(winner, f)


def test_ai(config):
    width, height = 700, 500
    window = pygame.display.set_mode((width, height))
    # Load the winner
    with open("best.pickle2", "rb") as f:
        winner = pickle.load(f)
    game = PongGame(window, width, height)
    game.test_ai(winner, config)


if __name__ == "__main__":
    loca_dir = os.path.dirname(__file__)
    config_path = os.path.join(loca_dir, "config.txt")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    # run_neat(config)
    test_ai(config)
