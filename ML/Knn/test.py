import pandas as pd
import math
import matplotlib.pyplot as plt
import copy
data=pd.read_csv("semeion.csv",header=None)
colum=list(data.iloc[0,:])
colum=[colum[i] for i in range(257)]
print(type(colum))
print(colum)
print(len(colum))
data=pd.read_csv("semeion.csv")
data = data.values.tolist()
print(len(data[0]))
feature=[data[i][:-10] for i in range(0,len(data))]
target=[data[i][-10:] for i in range(0,len(data))]
target=[i.index(1) for i in target]
print(target)
print(len(target))
for i,j in enumerate(feature):
    j.append(target[i])
print(len(feature[0]))
newCsv=pd.DataFrame(columns=colum,data=feature)
newCsv.to_csv('new_Semeion.csv')
