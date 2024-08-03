# AI plays Ping Pong 
This project demonstrates training an AI to play a Pong game using the **NEAT (NeuroEvolution of Augmenting Topologies)** algorithm. 

The AI learns to control the paddles and play against itself, and humans, improving over generations.

## Description
This project entails the following:
1. Pong game implementation using **pygame**
2. AI training using NEAT algorithm
3. Visualization of AI playing pong game.
4. Saving and loading the best trained model.
5. Playing the pong game against the best AI or a friend.

## Prerequisites
Before you begin, ensure you have met the following requirements:
1. Python 3.7+
2. Pygame
3. NEAT-Python
4. Pickle

## Installation
1. Clone the Repository:
```bash
git clone https://github.com/NemesisXVI/Ping-Pong-NEAT.git
cd Ping-Pong-NEAT
```
2. Install required Packages:
```bash
pip install -r requirements.txt
```

## Usage
### 1. Try out the Game
Run the file **'Play1v1.py'** and play with a friend or go at it completely yourself.
```python
python Play1v1.py
```
The controls for the Paddles are:

1. 'W' : Move Left Paddle Up
2. 'S' : Move Left Paddle Down
3. 'Up Arrow' : Move Right Paddle Up
4. 'Down Arrow' : Move Right Paddle Down

The Game Looks like this:
![alt text](E:\Python\Ping-Pong-NEAT\GameWIndow.png)

### 2. Train AI
Run the file **'main.py'** to start training the AI using the NEAT algorithm.
You can observe the statistics per generation in the console.
```python
python main.py
```

### 3. Play Against the AI
To play against the trained AI model, run the file **'Play_AI.py'** to start the game.
```python
python Play_AI.py
```

## Understanding the NEAT Algorithm
NEAT (NeuroEvolution of Augmenting Topologies) is an evolutionary algorithm that generates artificial neural networks (ANNs).
NEAT evolves both the topology and the weights of neural networks, allowing it to discover both the structure and the parameters of effective neural networks.
The key concepts of NEAT are:

### 1. Genomes and Phenotypes:

* #### Genome: 
   A representation of a neural network, consisting of nodes (neurons) and connections (synapses).
* #### Phenotype: 
  The actual neural network created from the genome, which can be evaluated in the given task.

### 2. Speciation:
NEAT groups similar genomes into species to protect innovative structures during the early stages of evolution. This prevents premature convergence to suboptimal solutions.

### 3. Crossover and Mutation:

* #### Crossover:
  Combines the genes of two parent genomes to produce offspring. This helps to combine beneficial traits from different parents.
* #### Mutation: 
  Alters genomes by adding nodes or connections, or by modifying the weights of existing connections. This introduces new structures and behaviors into the population.
### 4. Fitness Sharing:

 Within each species, individuals share fitness to reduce competition among closely related individuals. This ensures diversity and maintains multiple solutions in parallel.
### 5. Incremental Growth:

NEAT starts with simple networks and gradually complexifies them by adding nodes and connections over generations. This incremental growth allows NEAT to optimize both simple and complex solutions.

## Further on the NEAT 
Check out the following for more information on NEAT algorithm:
1. Original paper by Kenneth O. Stanley and Risto Miikkulainen : https://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf
2. Article by Hannah Han L. : https://www.linkedin.com/pulse/neuroevolution-neat-algorithm-my-hannah-le/
3. Article by Robert MacWha : https://macwha.medium.com/evolving-ais-using-a-neat-algorithm-2d154c623828
4. NEAT-python Documentation: https://neat-python.readthedocs.io/en/latest/neat_overview.html
5. Application of NEAT by Tech with Tim to create Flappy Bird Game : Flappy Bird with Neat : https://www.youtube.com/playlist?list=PLzMcBGfZo4-lwGZWXz5Qgta_YNX3_vLS2

## Implementation of NEAT in Pong
### Setting up the config file
1. **Population Size** : Setting population per generation to 50.
2. **Fitness Threshold** : To stop the algorithm once a genome attains a fitness of 400.
3. **Fitness Criteria** : Set to max, implying that when a single genome crosses the fitness threshold, the algorithm stops and outputs the genome.
4. **Activation Default** : I have chosen relu to get an output which is greater than 0.
5. **Hidden Layers** : After trial and error I came upto the value of 2.
6. **Number of Inputs** : Set to 3.
7. **Number of Outputs** : Set to 3.

These are the most important parameters in the config file, do check out the other parameters as well for better understanding.
### Inputs and Outputs
#### Inputs:
I am giving 3 inputs to the NEAT algorithm for each genome which are as follows:
* y coordinate of paddle
* y coordinate of ball
* absolute difference between the x coordinates of the paddle and the ball.

#### Outputs:
I am receiving 3 outputs from every genome which are as follows:
* Probability to not move the paddle
* Probability to move the paddle up 
* Probability to move the paddle down.

### Moving the Paddles
After receiving the outputs in the form of probabilities to move the paddle, I find the maximum of the 3 outputs and if
it is possible execute that motion.
``` python
output = net.activate((self.right_paddle.y, self.ball.y, abs(self.right_paddle.x - self.ball.x)))
            decision = output.index(max(output))
if decision == 0:
    pass
elif decision == 1:
    self.game.move_paddle(left=False, up=True)
else:
    self.game.move_paddle(left=False, up=False)

```

### Calculating Fitness
After trying various fitness functions, I settled on the function which calculates the fitness as the number of times the 
genome actually manages to hit the ball.
``` python
    def calculate_fitness(self, genome1, genome2, game_info):
        genome1.fitness += game_info.left_hits
        genome2.fitness += game_info.right_hits
```
The more the no. of hits, more is the fitness, so the algorithm is forced to hit the ball more no. of times thereby
learning to always avoid missing the ball.
For each game that the genome plays, the fitness is incremented by the no. of times the genome hits the ball in that game.

### Training the Genome:
For the genome to actually learn to hit the ball, it needs to play with an opponent.
So instead of manually sitting and playing with each genome, I make them play against each other.
This is **NOT** the best way to approach the problem but it is the best I could come up with.

This is implemented by fixing the left paddle with a genome and iterating over all the other genomes in the population 
to play as the right paddle.
Each game ends when either one of the genomes misses the ball.
``` python
    for i, (genome_id1, genome1) in enumerate(genomes):
        if i == len(genomes) - 1:
            break
        genome1.fitness = 0
        for genome_id2, genome2 in genomes[i + 1:]:
            if genome2.fitness == None:
                genome2.fitness = 0
            game = PongGame(window, width, height)
            game.train_ai(genome1, genome2, config)
```
The biggest drawback of this method is that a genome is only as good as it's opponent.
So it could so happen that the left paddle does not miss a single time but the right paddle
misses on the first try everytime, thereby reducing the overall fitness of the left paddle genome.

### Saving the Winner
I have used **pickle** to store the first genome to cross the fitness threshold.
This pickle file can they be used to play against.

## Support
For any queries regarding the NEAT or pygame implementation, contact me through my socials.
