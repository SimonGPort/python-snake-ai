import random
import math
import torch
import numpy as np
from collections import deque
from snake import SnakeGameAI

MAX_MEMORY=100_000
BATCH_SIZE=1000
LR=0.001

class Agent:
    def __init__(self):
        self.n_games=0
        self.epsition=0
        self.gamma=0
        self.memory=deque(maxlen=MAX_MEMORY) #popleft
        # TODO: model,trainer


    def get_state(self,game):
        dangerStraight=0
        dangerRight=0
        dangerLeft=0

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
            if game.direction=='DOWND':
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
        








    def remember(self,state,action,reward,next_state,done):
        pass
    def train_long_memory(self):
        pass
    def train_short_memory(self,state,action,reward,next_state,done):
        pass
    def get_action(self,state):
        pass

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
        reward,done,score,=game.play_step(final_move)
        state_new=agent.get_state(game)

        #train short memory
        agent.train_short_memory(state_old,final_move,reward,state_new,done)

        #remember
        agent.remember(state_old,final_move,reward,state_new,done)

        if done:
            #train long memory, plot result
            game.reset()
            agent.n_game+=1
            agent.train_long_memory()

            if score > record:
                record=score
            print('Game:',agent.n_games,'Score:',score,'Record:',record)

            #todo mathplotlib graph



if __name__=='__main__':
    train()