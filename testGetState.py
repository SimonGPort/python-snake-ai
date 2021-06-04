import math

def get_state(game):
        dangerStraight=0
        dangerRight=0
        dangerLeft=0

        firstCol=[1,33,65,97,129,161,193,225,257,289,321,353,385,417,449,481,513,545,577,609,641,673,705,737]
        lastCol=[32,64,96,128,160,192,224,256,288,320,352,384,416,448,480,512,544,576,608,640,672,704,736,768]
        topCol=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
        downCol=[737,738,739,740,741,742,743,744,745,746,745,746,747,748,749,750,751,752,753,754,755,756,757,758,759,760,761,762,763,764,765,766,767,768]
        #find danger with the border of the playground
        if game['body'][0] in topCol:
            if game['direction']=='UP':
                dangerStraight=1
            if game['direction']=='RIGHT':
                dangerLeft=1
            if game['direction']=='LEFT':
                dangerRight=1
        if game['body'][0] in downCol:
            if game['direction']=='DOWN':
                dangerStraight=1
            if game['direction']=='RIGHT':
                dangerRight=1
            if game['direction']=='LEFT':
                dangerLeft=1
        if game['body'][0] in firstCol:
            if game['direction']=='LEFT':
                dangerStraight=1
            if game['direction']=='UP':
                dangerLeft=1
            if game['direction']=='DOWN':
                dangerRight=1
        if game['body'][0] in lastCol:
            if game['direction']=='RIGHT':
                dangerStraight=1
            if game['direction']=='UP':
                dangerRight=1
            if game['direction']=='DOWN':
                dangerLeft=1

        #find the danger with his body
        StraightSide=None
        RightSide=None
        LeftSide=None

        if game['direction']=='RIGHT':
            StraightSide=game['body'][0]+1
            RightSide=game['body'][0]+33
            LeftSide=game['body'][0]-33
        if game['direction']=='LEFT':
            StraightSide=game['body'][0]-1
            RightSide=game['body'][0]-33
            LeftSide=game['body'][0]+33
        if game['direction']=='UP':
            StraightSide=game['body'][0]-33
            RightSide=game['body'][0]+1
            LeftSide=game['body'][0]-1
        if game['direction']=='DOWN':
            StraightSide=game['body'][0]+33
            RightSide=game['body'][0]-1
            LeftSide=game['body'][0]+1

        if StraightSide in game['body']:
            dangerStraight=1
        if RightSide in game['body']:
            dangerRight=1
        if LeftSide in game['body']:
            dangerLeft=1
        
        directionLeft=0
        directionRight=0
        directionUp=0
        directionDown=0

        if game['direction']=='UP':
            directionUp=1
        if game['direction']=='DOWN':
            directionDown=1
        if game['direction']=='LEFT':
            directionLeft=1
        if game['direction']=='RIGHT':
            directionRight=1

        foodLeft=0
        foodRight=0
        foodUp=0
        foodDown=0

        rowFood=math.floor(game['food']/32)
        if game['food']%32==0:
            rowFood-=1

        rowHead=math.floor(game['body'][0]/32)
        if game['food']%32==0:
            rowHead-=1
        
        if rowFood>rowHead:
            foodUp=1
        if rowFood<rowHead:
            foodUp=1
        colFood=game['food']-(rowFood*32)
        if colFood==0:
            colFood=32
        colHead=game['body'][0]-(rowHead*32)
        if colHead==0:
            colHead=32
        if colFood>colHead:
            foodRight=1
        if colFood<colHead:
            foodLeft=1

        state=[dangerStraight,dangerRight,dangerLeft,directionLeft,directionRight,directionUp,directionDown,foodLeft,foodRight,foodUp,foodDown]
        print('state:',state)
        
x={'direction':"DOWN",'food':768,'body':[384]}
get_state(x)