import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import seaborn as sns

import random


def calcDataSetAgeDistribution(path):

    images = os.listdir(path)
    data = []
    for img in images:
        splitedFileName = img.split("-")
        for part in splitedFileName:
            if part.startswith("N"):
                year = int(part.strip("N"))
                data.append(year)

    plt.hist(data, bins=[14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
    plt.ylabel("Occurences")
    plt.xlabel("year")
    plt.title("Occurences by year")
    plt.show()

def plotColorDistributionHistogramm(img):
    
    img = cv2.imread(img)
    color = ('b','g','r')
    fig = plt.figure()
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    return fig


   



def plotColorDistributionHistogramms(imgFolder):
    for img in os.listdir(imgFolder):
        imgPath = os.path.join(imgFolder,img)
        fig = plotColorDistributionHistogramm(imgPath)
        outpath = os.path.join(r'C:\Users\felix\Desktop\Neuro\histogramms',img.strip(".jpg")+"histogramm.jpg")
        fig.savefig(outpath, dpi=fig.dpi)
    return



def greyScaleIntesityOverDataSet(imageFolder):

    means = []

    for image in os.listdir(imageFolder):
        imagePath = os.path.join(imageFolder,image)
        img_rgb = cv2.imread(imagePath)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        meanVal = img_gray.mean(axis=0).mean()

        means.append(meanVal)


    sns.set_style('whitegrid')
    plot = sns.kdeplot(np.array(means))
    fig = plot.get_figure()
    fig.savefig(r'C:\Users\felix\Desktop\Neuro\histogramms\GreyScaleHisto.png', dpi=fig.dpi)
    plt.show()
    return


def meanColorPlotOverDataSet(imageFolder):
    nb_bins = 256
    count_r = np.zeros(nb_bins)
    count_g = np.zeros(nb_bins)
    count_b = np.zeros(nb_bins)
    fig = plt.figure()
    for image in os.listdir(imageFolder):
      
        if not image.endswith(".jpg"):
            continue  
        imgPath = os.path.join(imageFolder,image)
        img = Image.open(imgPath)
        x = np.array(img)
        x = x.transpose(2, 0, 1)
        hist_r = np.histogram(x[0], bins=nb_bins, range=[0, 255])
        hist_g = np.histogram(x[1], bins=nb_bins, range=[0, 255])
        hist_b = np.histogram(x[2], bins=nb_bins, range=[0, 255])
        count_r += hist_r[0]
        count_g += hist_g[0]
        count_b += hist_b[0]

    bins = hist_r[1]
    
    plt.bar(bins[:-1], count_r, color='r', alpha=0.7)
    plt.bar(bins[:-1], count_g, color='g', alpha=0.7)
    plt.bar(bins[:-1], count_b, color='b', alpha=0.7)
    
    fig.savefig(r'C:\Users\felix\Desktop\Neuro\histogramms\RGBHisto.png', dpi=fig.dpi)
    plt.show()

#creates a scatterplot of the distribution of the 
def scatterPlotRGBDistribution(imgFolder):

    rVals = []
    gVals = []
    bVals = []

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')

    for img in os.listdir(imgFolder):
        imgPath = os.path.join(imgFolder,img)

        image = cv2.imread(imgPath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        avgColorPerRow = np.average(image, axis=0)
        avgColor = np.average(avgColorPerRow, axis=0)
        
    
        rVals.append(avgColor[0])
        gVals.append(avgColor[1])
        bVals.append(avgColor[2])

    ax.scatter(rVals, gVals, bVals)
    ax.set_xlabel('red')
    ax.set_ylabel('green')
    ax.set_zlabel('blue')
    fig.savefig(r'C:\Users\felix\Desktop\Neuro\histogramms\Scatter.png', dpi=fig.dpi)
    plt.show()

    return

def plotColorDensity(path):
    return


if __name__ == '__main__':
    #scatterPlotRGBDistribution(r"F:\split")
    #greyScaleIntesityOverDataSet(r"F:\split")
    #meanColorPlotOverDataSet(r"F:\split")
    plotColorDistributionHistogramms(r"F:\split")


