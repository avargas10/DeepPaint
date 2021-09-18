
# importing the required module
import matplotlib.pyplot as plt

import csv


def graphLossResults(file):
    epoch = []
    loss = []
    with open(file,'rt')as f:
        data = csv.reader(f)
        for row in data:
            epoch.append(row[0])
            loss.append(row[1])
        # plotting the points
        plt.plot(epoch, loss)
        
        # naming the x axis
        plt.xlabel('x - axis')
        # naming the y axis
        plt.ylabel('y - axis')
        
        # giving a title to my graph
        plt.title('My first graph!')
        
        # function to show the plot
        plt.show()


graphLossResults('C:/Users/VargasKiller/Documents/Proyects/DeepPaint/T0104_DeepPaint_SmootL1.csv')