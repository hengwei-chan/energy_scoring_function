import numpy as np, pandas as pd

def readfile(jobfile,type):
    inputfile = open(jobfile, "r")
    if type=='t':
        text = inputfile.read()
    if type=='l':
        text = inputfile.readlines()
    inputfile.close()
    return text

def distance(i, j):
    d = ((i[0] - j[0]) ** 2 + (i[1] - j[1]) ** 2 + (i[2] - j[2]) ** 2) ** 0.5
    return d

