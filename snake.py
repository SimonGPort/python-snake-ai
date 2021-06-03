import pygame
import random
import math

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

# ---constantes
BLOCK_SIZE=20

SPEED=60

# rgb colors
WHITE=(255,255,255)
RED=(200,0,0)
BLUE1=(0,0,255)
BLUE2=(0,100,255)
BLACK=(0,0,0)



class SnakeGameAI:
    def __init__(self,w=640,h=480):
        self.game_over=False
        self.w=w
        self.h=h
        self.display=pygame.display.set_mode((self.w,self.h))
        pygame.display.set_caption("Snake")
        self.clock=pygame.time.Clock()
        self.reset()

    def placeFood(self):
        x=random.randint(1,32)
        y=random.randint(1,24)
        foodPosition=x*y
        if foodPosition in self.body:
            self.placeFood()
        else:
            self.food=foodPosition

    def play_step(self,action):
        self.frame_iteration+=1

        #1 collect user input
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                pygame.quit()
                quit()

        #2 move
        self.move(action)
        #3 check if game over
        
        if self.collision():
            self.game_over=True
        if self.game_over==True or self.frame_iteration > 100 * len(self.body):
            self.reward= -10

        #4 place new food or just move
        # self.placeFood()
        if self.body[0]==self.food:
            DirectionTailValue=self.body[-1]-self.body[-2]

            if DirectionTailValue==-1:
                # DirectionTail='LEFT'
                self.body.append(self.body[-1]-1)
                self.food=None
                self.placeFood()
                self.score +=1
                self.reward+=10
            if DirectionTailValue==1:
                # DirectionTail='RIGHT'
                self.body.append(self.body[-1]+1)
                self.food=None
                self.placeFood()
                self.score +=1
                self.reward+=10
            if DirectionTailValue==-32:
                # DirectionTail='TOP'
                self.body.append(self.body[-1]-32)
                self.food=None
                self.placeFood()
                self.score +=1
                self.reward+=10
            if DirectionTailValue==32:
                # DirectionTail='DOWN'
                self.body.append(self.body[-1]+32)
                self.food=None
                self.placeFood()
                self.score +=1
                self.reward+=10

        

        #5 update ui and clock
        self.updateUi()
        self.clock.tick(SPEED)

        #6 return game over and score
        
        return self.reward, self.game_over, self.score

    def reset(self):
        # init game state
        self.score=0
        # l'array des lives represente la position de chanque morceau du corps, a l'index 0 il y a la tete
        self.body=[368,367,366]
        self.direction='RIGHT'
        self.food=None
        self.placeFood()
        self.frame_iteration=0
        self.reward=0
        self.game_over=False

    def collision(self):
        #hit itself
        return self.body[0] in self.body[1:]

    def move(self,action):
        #[straight,right,left]
        #conversion de action vers l'axe direction
        if action[0]==1:
            direction=self.direction
        if action[1]==1:
            if self.direction=='UP':
                direction='RIGHT'
            if self.direction=='RIGHT':
                direction='DOWN'
            if self.direction=='DOWN':
                direction='LEFT'
            if self.direction=='LEFT':
                direction='UP'
        if action[2]==1:
            if self.direction=='UP':
                direction='LEFT'
            if self.direction=='LEFT':
                direction='DOWN'
            if self.direction=='DOWN':
                direction='RIGHT'
            if self.direction=='RIGHT':
                direction='UP'
        self.direction=direction

        body=self.body.copy()

        #verification si le mouvement est a l'exterieur
        firstCol=[1,33,65,97,129,161,193,225,257,289,321,353,385,417,449,481,513,545,577,609,641,673,705,737]
        lastCol=[32,64,96,128,160,192,224,256,288,320,352,384,416,448,480,512,544,576,608,640,672,704,736,768]
        topCol=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
        downCol=[737,738,739,740,741,742,743,744,745,746,745,746,747,748,749,750,751,752,753,754,755,756,757,758,759,760,761,762,763,764,765,766,767,768]

        if self.direction=='RIGHT' and self.body[0] in lastCol:
            self.game_over=True
            return
        if self.direction=='LEFT' and self.body[0] in firstCol:
            self.game_over=True
            return
        if self.direction=='UP' and self.body[0] in topCol:
            self.game_over=True
            return
        if self.direction=='DOWN' and self.body[0] in downCol:
            self.game_over=True
            return

        if self.direction=='RIGHT':
            body[0]=body[0]+1
        if self.direction=='LEFT':
            body[0]=body[0]-1
        if self.direction=='UP':
            body[0]=body[0]-32
        if self.direction=='DOWN':
            body[0]=body[0]+32

        for index, bodyPart in enumerate(self.body):
            if index != 0:
                body[index]=self.body[index-1]
        self.body=body

    def updateUi(self):
        self.display.fill(BLACK)
        for point in self.body:
            row=math.floor(point/32)
            if point%32==0:
                row-=1
            xPosition=(point-row*32-1)*BLOCK_SIZE
            yPosition=row*BLOCK_SIZE
            pygame.draw.rect(self.display,BLUE1,pygame.Rect(xPosition,yPosition,BLOCK_SIZE,BLOCK_SIZE))
            pygame.draw.rect(self.display,BLUE2,pygame.Rect(xPosition+4,yPosition+4,12,12))

        rowFood=math.floor(self.food/32)
        if self.food%32==0:
            rowFood-=1
        xPositionFood=(self.food-rowFood*32-1)*BLOCK_SIZE
        yPositionFood=rowFood*BLOCK_SIZE

        pygame.draw.rect(self.display,RED,pygame.Rect(xPositionFood,yPositionFood,BLOCK_SIZE,BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.update()
