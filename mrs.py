#import random

import numpy as np
#import pandas as pd
import webbrowser
import sys
import time


#（列名 1-悲伤，2-安静，3-治愈，4-放松，5-愉快，6-令人激动， 行名 1-卧室，2-客厅，3-办公地点，4-浴室）
lm = [0,0,0,0,0,0,0,
     0,0,0.25,0.15,0.4,0.2,0,
     0,0,0.14,0.13,0.28,0.32,0.13,
     0,0,0.28,0,0.4,0.32,0,
     0,0,0,0,0.6,0.4,0]
#（列名 1-悲伤，2-安静，3-治愈，4-放松，5-愉快，6-令人激动， 行名 1-愤怒，2-厌恶，3-恐惧，4-开心，5-悲伤，6-惊讶，7-自然）
pm = [0,0,0,0,0,0,0,
     0,0,0.33,0.33,0.34,0,0,
     0,0,0.3,0.3,0.4,0,0,
     0,0,0.3,0.25,0.45,0,0,
     0,0,0,0,0.29,0.44,0.27,
     0,0.2,0.2,0.35,0.25,0,0,
     0,0,0,0.33,0.28,0.39,0,
     0,0,0.2,0,0.35,0.25,0.2,]
tm = [0,0,0,0,0,0,0,
     0,0.08,0.2,0.06,0.31,0.33,1,
     0,0.07,0.26,0.05,0.26,0.26,0.1,
     0,0.06,0.2,0.05,0.25,0.34,0.1,
     0,0.12,0.33,0.1,0.2,0.2,0.05]
MM = [0,0.1,0.3,0.45,0.6,0.8,0.95]
#C = [0,0.2,0.45,0.55,0.65,0.775,0.925]
#A表示地点和歌曲情绪之间的对应矩阵4*7，不能用0检索（数根据调研结果改）
LM = np.array(lm).reshape(5,7)
#B表示人的情绪和歌曲情绪之间的对应矩阵7*7，不能用0检索（数根据调研结果改）
PM = np.array(pm).reshape(8,7)
TM = np.array(tm).reshape(5,7)
def get_location():
    print("输入地点序号：")
    i = input()
    return i

def get_personemo():
    print("输入人的情绪序号：")
    i = input()
    return i

def get_timeorder():
    #时间分区间，共4个区间
    t = time.localtime()
    i = t.tm_hour
    if i>=5 and i<11:
        a = 1
    elif i>=11 and i<17:
        a = 2
    elif i>=17 and i<23:
        a = 3
    else :
        a = 4
    #print(a)
    return a

def get_lo_score(location):
    #地点→音乐情绪
    c = []
    d = []
    for i in range(1, 7):
        if (LM[int(location)][int(i)] != 0):
            c.append(i)
            d.append(LM[int(location)][int(i)])
    #print(c)
    #print(d)
    #np.random.seed(i)
    p = np.array(d)
    index = np.random.choice(c, p=p.ravel())
    #print(index)
    resultlo = MM[index]
    return resultlo

def get_personemo_score(personemo):
    #情绪→音乐情绪
    c = []
    d = []
    for i in range(1, 7):
        if (PM[int(personemo)][int(i)] != 0):
            c.append(i)
            d.append(PM[int(personemo)][int(i)])
    #print(c)
    #print(d)
    #np.random.seed(i)
    p = np.array(d)
    index = np.random.choice(c, p = p.ravel())
    #print(index)
    resultemo = MM[index]
    return resultemo

def get_time_score(time):
    #时间→音乐情绪
    c = []
    d = []
    for i in range(1, 7):
        if (TM[int(time)][int(i)] != 0):
            c.append(i)
            d.append(TM[int(time)][int(i)])
    #print(c)
    #print(d)
    #np.random.seed(i)
    p = np.array(d)
    index = np.random.choice(c, p=p.ravel())
    #print(index)
    resultlo = MM[index]
    return resultlo


def record(lo_score,personemo_score,time_score):
    #计算最终要推荐的情绪评分
    weight = [0.2,0.7,0.1] #根据实验结果
    #score = lo_score * float(weight[0]) + personemo_score * float(weight[1]) + time_score * float(weight[2])
    score = lo_score * 0.2 + personemo_score * 0.7 + time_score * 0.1
    return score
def Judge(score):
    #具体score评级与上面相同
    #非正常情况判断
    if score<=0:
        print("fail1")
        print(score)
        sys.exit(0)
    elif score>=1:
        print("fail1")
        sys.exit(0)
    #正常情况
    else :
        if score<0.2 :
            return "悲伤"
        elif score>=0.2 and score<0.4 :
            return "安静"
        elif score>=0.4and score<0.5 :
            return "治愈"
        elif score>=0.5 and score<0.7 :
            return "放松"
        elif score>=0.7 and score<0.9 :
            return "愉快"
        else :
            return "令人激动的"
def flavor(location,emotion,time): #传的参数待定
    #如果查到了返回查到的音乐
    #MS查表推荐
    return "流行"
def search(emo,style):
    webbrowser.open("https://y.qq.com/n/ryqq/search?w={}".format(emo + " " +style))



def main():
    lo = get_location()
    #print(lo)
    person_emo = get_personemo()
    time1 = get_timeorder()
    #print(time1)
    lo_score = get_lo_score(lo)
    print('LO:',lo_score)
    personemo_score = get_personemo_score(person_emo)
    print('PO:',personemo_score)
    time_score = get_time_score(time1)
    print('TO:',time_score)
    score1 = record(lo_score,personemo_score,time_score)
    print(score1)
    music_emo = Judge(score1)
    print(music_emo)
    style1 = flavor(lo,person_emo,time1)
    print(style1)
    #search(music_emo,style1)


main()

# i = get_personemo_score(2)
# print(i)
