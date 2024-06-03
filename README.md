# Reinforcement Learning Agents for the Game of Differenzler Jass
This project tries to apply DeepQ learning to the Differenzler Jass using [RLCard](https://github.com/datamllab/rlcard). 

## Setup
The following packages are required to run the project:
 - PyTorch
 - Numpy
 - RLCards
 - Pandas
 - Matplotlib

## Running the project
### Train an agent
To train an agent use the following script:
```console
train_agent.py -m src/diff -o log -e 1000 --state-rep compressed --prediction rand
```
More details can be found using
```console
train_agent.py -h
```

### Evaluating an agent
To evaluate an agent use the following script:
```console
evaluate_agent.py -m src/diff -t dqn -w path/to/model/weights/model.pth --eval-games 1000
```
More details can be found using
```console
evaluate_agent.py -h
```

### Visualize training performance
To visualize the performance of the agent during training use the following script:
```console
plot_training_curves.py --data path/to/training/out/floder --title "Compressed State" --show true
```
More details can be found using
```console
plot_training_curves.py -h
```
