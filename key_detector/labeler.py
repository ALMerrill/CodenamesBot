#!/usr/bin/env python
import numpy as np
import sys,os
import csv
import cv2
# import matplotlib.pyplot as plt
class Labeler:
    def __init__(self):
        self.card=np.zeros((5,5),dtype=int)
        self.scaled=np.zeros((400,400,3))
        self.x=0
        self.y=0
        
    def get_color(self,val):   
        RED=(0,0,255)
        BLUE=(255,0,0)
        BLACK=(0,0,0)
        MAGENTA=(255,255,0)
        TAN=(140,180,210)
        color=TAN
        if val == 0:
            color=TAN
        elif val == 1:
            color =RED
        elif val == 2:
            color = BLUE
        elif val ==3:
            color = BLACK
        else:
            color = MAGENTA
        return color
    def r(self):
        f = open("input.txt","r")
        contents=f.read()
        for row,line in enumerate(contents.split("\n")):
            for char in range(5):
                self.card[row][char]=line[int(char)]
        f.close()
    def rl(self):
        self.card=np.rot90(self.card)
        self.p()
    def rr(self):
        self.card=np.rot90(self.card,axes=(1,0))
        self.p()
    def get_label(self):
        card=self.card.flatten()
        card=np.array2string(card, separator=',')[1:-1]
        id_=self.get_next_label_id()
        return "{},{}\n".format(id_,card)

    def set_scaled(self):
        for row in range(5):
            for col in range(5):
                val= self.card[row,col]
                color = self.get_color(val)
                
                self.scaled[row*80:(row+1)*80,col*80:(col+1)*80]=color

    def w(self):
        with open('labels.csv','a') as fd:
            fd.write(self.get_label())
    
    def get_next_label_id(self):
        with open('labels.csv', 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            label=-1
            for row in reader:
                temp=int(row[0])
                if label<temp:
                    label=temp
            return str(label+1).zfill(5)

    def p(self):
        # self.card[2,2]=2
        self.set_scaled()
        # self.highlight(self.y,self.x)
        cv2.imshow('test',self.scaled)
        cv2.waitKey(0)

    def relabel(self):
        with open('labels.csv','rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='\'')
            with open('label2.csv','a') as fd:
                for i,row in enumerate(reader):
                    
                    temp=str(i).zfill(5)
                    t2=row[1:]
                    t2=np.array(t2,dtype=int)

                    
                    
                    fd.write("{},{}\n".format(temp,np.array2string(t2,separator=',')[1:-1]))

l=Labeler()
l.relabel()
