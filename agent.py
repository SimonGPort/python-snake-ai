import random
import math
import torch
import numpy as np
from collections import deque
from snake import SnakeGameAI
from model import Linear_Qnet, QTrainer
from helper import plot

MAX_MEMORY=100_000
BATCH_SIZE=1000
LR=0.001

class Agent:
    def __init__(self):
        self.n_games=0
        self.epsition=0
        self.gamma=0.9
        self.memory=deque(maxlen=MAX_MEMORY) #popleft
        self.model=Linear_Qnet(11,256,3)
        self.trainer=QTrainer(self.model,lr=LR,gamma=self.gamma)


    def get_state(self,game):
        dangerStraight=0
        dangerRight=0
        dangerLeft=0
        print("direction:",game.direction)

        firstCol=[1,33,65,97,129,161,193,225,257,289,321,353,385,417,449,481,513,545,577,609,641,673,705,737]
        lastCol=[32,64,96,128,160,192,224,256,288,320,352,384,416,448,480,512,544,576,608,640,672,704,736,768]
        topCol=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
        downCol=[737,738,739,740,741,742,743,744,745,746,745,746,747,748,749,750,751,752,753,754,755,756,757,758,759,760,761,762,763,764,765,766,767,768]
        #find danger with the border of the playground
        if game.body[0] in topCol:
            if game.direction=='UP':
                dangerStraight=1
            if game.direction=='RIGHT':
                dangerLeft=1
            if game.direction=='LEFT':
                dangerRight=1
        if game.body[0] in downCol:
            if game.direction=='DOWN':
                dangerStraight=1
            if game.direction=='RIGHT':
                dangerRight=1
            if game.direction=='LEFT':
                dangerLeft=1
        if game.body[0] in firstCol:
            if game.direction=='LEFT':
                dangerStraight=1
            if game.direction=='UP':
                dangerLeft=1
            if game.direction=='DOWN':
                dangerRight=1
        if game.body[0] in lastCol:
            if game.direction=='RIGHT':
                dangerStraight=1
            if game.direction=='UP':
                dangerRight=1
            if game.direction=='DOWN':
                dangerLeft=1

        #find the danger with his body
        StraightSide=None
        RightSide=None
        LeftSide=None

        if game.direction=='RIGHT':
            StraightSide=game.body[0]+1
            RightSide=game.body[0]+33
            LeftSide=game.body[0]-33
        if game.direction=='LEFT':
            StraightSide=game.body[0]-1
            RightSide=game.body[0]-33
            LeftSide=game.body[0]+33
        if game.direction=='UP':
            StraightSide=game.body[0]-33
            RightSide=game.body[0]+1
            LeftSide=game.body[0]-1
        if game.direction=='DOWN':
            StraightSide=game.body[0]+33
            RightSide=game.body[0]-1
            LeftSide=game.body[0]+1

        if StraightSide in game.body:
            dangerStraight=1
        if RightSide in game.body:
            dangerRight=1
        if LeftSide in game.body:
            dangerLeft=1
        
        directionLeft=0
        directionRight=0
        directionUp=0
        directionDown=0

        if game.direction=='UP':
            directionUp=1
        if game.direction=='DOWN':
            directionDown=1
        if game.direction=='LEFT':
            directionLeft=1
        if game.direction=='RIGHT':
            directionRight=1

        foodLeft=0
        foodRight=0
        foodUp=0
        foodDown=0

        rowFood=math.floor(game.food/32)
        if game.food%32==0:
            rowFood-=1

        rowHead=math.floor(game.body[0]/32)
        if game.food%32==0:
            rowHead-=1
        
        if rowFood>rowHead:
            foodUp=1
        if rowFood<rowHead:
            foodUp=1
        colFood=game.food-(rowFood*32)
        if colFood==0:
            colFood=32
        colHead=game.body[0]-(rowHead*32)
        if colHead==0:
            colHead=32
        if colFood>colHead:
            foodRight=1
        if colFood<colHead:
            foodLeft=1

        state=[dangerStraight,dangerRight,dangerLeft,directionLeft,directionRight,directionUp,directionDown,foodLeft,foodRight,foodUp,foodDown]
        return np.array(state,dtype=int)





    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))

    def train_long_memory(self):
        if len(self.memory)>BATCH_SIZE:
            mini_sample=random.sample(self.memory,BATCH_SIZE)
        else:
            mini_sample=self.memory
        states,actions,rewards,next_states,dones = zip(*mini_sample)
        self.trainer.train_step(states,actions,rewards,next_states,dones)

    
    def train_short_memory(self,state,action,reward,next_state,done):
        self.trainer.train_step(state,action,reward,next_state,done)


    def get_action(self,state):
        #random moves: tradeoff exploration / exploitation
        self.epsilon=80 - self.n_games
        final_move=[0,0,0]
        if random.randint(0,200)<self.epsilon:
            move=random.randint(0,2)
            final_move[move]=1
        else:
            state0=torch.tensor(state,dtype=torch.float)
            prediction=self.model(state0)
            move=torch.argmax(prediction).item()
            final_move[move]=1
        return final_move


def train():
    plot_scores=[]
    plot_mean_scores=[]
    total_score=0
    record=0
    agent=Agent()
    game=SnakeGameAI()
    while True:
        #get old state
        state_old=agent.get_state(game)

        #get move
        final_move=agent.get_action(state_old)

        #perform move and get new state
        reward,done,score=game.play_step(final_move)
        state_new=agent.get_state(game)

        #train short memory
        agent.train_short_memory(state_old,final_move,reward,state_new,done)

        #remember
        agent.remember(state_old,final_move,reward,state_new,done)

        if done:
            #train long memory, plot result
            game.reset()
            agent.n_games+=1
            agent.train_long_memory()

            if score > record:
                record=score
                agent.model.save()

            print('Game:',agent.n_games,'Score:',score,'Record:',record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__=='__main__':
    train()