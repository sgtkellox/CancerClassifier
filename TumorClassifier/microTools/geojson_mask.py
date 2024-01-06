import fiona
import rasterio
import rasterio.mask

import matplotlib.pyplot as plt
import numpy as np

import cv2

import argparse

import os


    
def getSLidePath(json, pathToSlides):
    fileName = json.split(".")[0]
    slidePath = os.path.join(pathToSlides,fileName+".svs")
    print("now processing: " + fileName)
    return slidePath
    
    

def tileAnnotatedArea(slide,json, tilePath, tileSize):
    

    with fiona.open(json, "r") as geojson:
       features = [feature["geometry"] for feature in geojson]

    with rasterio.open(slide) as src:
       out_image, out_transform = rasterio.mask.mask(src, features, crop=True)
       out_meta = src.meta
       

    imageName = slide.split("\\")[-1].split(".")[0]

    out_image = np.moveaxis(out_image, 0, -1)

    x = 0
    y = 0

    while x < out_image.shape[1]-tileSize:
        y = 0
        while y < out_image.shape[0]-tileSize:

            tile = out_image[y:y+tileSize,x:x+tileSize]

            #tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)

            #number_of_black_pix = cv2.countNonZero(tile)
        
            number_of_black_pix  = np.sum(tile == (0,0,0))

            if not number_of_black_pix>10000:
                safePath = os.path.join(tilePath,imageName+"_"+str(x)+"_"+str(y)+".jpg")
                tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                cv2.imwrite(safePath,tile)
            y+=500
        x+=500
    
def tileAllJsons(slidePath, tilePath, jsonPath,tileSize):
    for json in os.listdir(jsonPath):
        if json.endswith(".geojson"):
            absjsonPath = os.path.join(jsonPath,json)
            correspondingSlide = getSLidePath(json,slidePath)
            tileAnnotatedArea(correspondingSlide,absjsonPath, tilePath, tileSize)
            
            


if __name__ == '__main__':

    argParser = argparse.ArgumentParser()

    argParser.add_argument("-w", "--wsis", help="The Path to the slides")
    argParser.add_argument("-t", "--tiles", help="the path where the tiles are supposed to be stored")
    argParser.add_argument("-j", "--jsons", help="the path to the folder containing the json files")
    argParser.add_argument("-s", "--size", help="tile size ")

    args = argParser.parse_args()

    slides = args.wsis
    tiles = args.tiles
    jsons = args.jsons
    tileSize  = args.size
    
    tileAllJsons(slides,tiles,jsons, tileSize)
    
    
    
    



