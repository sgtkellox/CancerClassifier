from genericpath import isfile
import json
from re import S
from openslide import OpenSlide
import os
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

import cv2
import numpy as np

import argparse

def extractNNumberFromJason(json):
    json = json.split(".")[0]
    split = json.split("-")
    nNumber = split[1]+"-"+split[2]
    return nNumber

def extractNNumberFromTile(tile):
    tile = tile.split("_")[0]
    split = tile.split("-")
    nNumber = split[1]+"-"+split[2]
    return nNumber

def getSLidePath(json, pathToSlides):
    fileName = json.split(".")[0]
    slidePath = os.path.join(pathToSlides,fileName+".svs")
    return slidePath

def getJsonPath(slide,pathToJson):
    fileName = slide.split(".")[0]
    jsonPath = os.path.join(pathToJson,fileName+".geojson")
    return jsonPath
    

def getSlideName(slide):
    slideName = slide.split("\\")[-1]
    slideName = slideName.split(".")[0]
    return slideName


def getProcessedImages(pathOut):
    images = set()
    
    for image in os.listdir(pathOut):
        imageName = extractNNumberFromTile(image)
        images.add(imageName)
        
    return list(images)

def geoJsonExists(slide,jsonFolder):
    jsonPath = getJsonPath(slide,jsonFolder)
    if os.path.isfile(jsonPath):
        print("Json found for : " + slide)
        return True
    else:
        return False
    return jsonPath
    

def tileAnnotatedArea(slidePath,geojson, tilePath, tileSize, level):
    

    with open(geojson) as f:
        geojson_data = json.load(f)

    # Extract the coordinates of the ROI
    
    for entry in geojson_data['features']:
        coordinates = entry['geometry']['coordinates'][0]

        # Create a polygon from the coordinates using the shapely library
        roi_polygon = Polygon(coordinates)

        # Open the whole-slide image using OpenSlide
    
        slide = OpenSlide(slidePath)

        # Define the size of the tiles to be extracted
        tile_size = (tileSize, tileSize)
    
    
        if level == 0:
            factor = 1
        elif level == 1:
            factor = 4
        elif level == 2:
            factor = 8
        
        grow = factor*  tileSize
    
        w0 = int(slide.properties["openslide.level[0].width"])
        h0 = int(slide.properties["openslide.level[0].height"])
    
        print(w0)
        print(h0)
    
   

        # Create the output directory if it doesn't exist
        os.makedirs(tilePath, exist_ok=True)

        # Iterate through the slide to extract tiles
        for x in range(0, w0-grow,grow ):
            for y in range(0, h0-grow, grow):
                # Read a region of the slide at the specified level and tile size
                tile = slide.read_region((x, y), level, tile_size)
            

                # Create a bounding box for the tile
                tile_bbox = Polygon(
                    [(x, y), (x + tile_size[0]*factor, y), (x + tile_size[0]*factor, y + tile_size[1]*factor), (x, y + tile_size[1]*factor)]
                )
                
                intersect = roi_polygon.intersection(tile_bbox).area
                
                percentage = intersect/tile_bbox.area
                
                print(percentage)
                    
                

                # Check if the tile intersects with the Region of Interest (ROI) polygon: add it to the list
                if tile_bbox.intersects(roi_polygon):
                   slideName = getSlideName(slidePath)
                   safePath = os.path.join(tilePath,slideName+"_"+str(int(x/factor))+"_"+str(int(y/factor))+".jpg")
                   tileRGB = tile.convert('RGB')
           
                   tileNP = np.array(tileRGB)
                   tileNP = cv2.cvtColor(tileNP, cv2.COLOR_BGR2RGB)
                   cv2.imwrite(safePath,tileNP)
           

    
    slide.close()
    
def tileAllJsons(slidePath, tilePath, jsonPath,tileSize):
    processedImages = getProcessedImages(tilePath)
    for json in os.listdir(jsonPath):
        
        if extractNNumberFromJason(json) in processedImages:
            print("item " + json + " allready processed")
            continue
        if json.endswith(".geojson"):
            absjsonPath = os.path.join(jsonPath,json)
            correspondingSlide = getSLidePath(json,slidePath)
            tileAnnotatedArea(correspondingSlide,absjsonPath, tilePath, tileSize, level)


    # Load the GeoJSON file

def tileKryos(slidePath, tilePath, jsonPath,tileSize, level, includedDiags):
    for slide in os.listdir(slidePath):       
        split = slide.split(".")[0].split("-")      
        if split[0] in includedDiags:
            
            if split[3] =="K":
                
                if split[4]=="G":
                    json = getJsonPath(slide,jsonPath)
                    filePath = os.path.join(slidePath,slide)
                    tileAnnotatedArea(filePath,json, tilePath, tileSize,level)









if __name__ == '__main__':
    
    argParser = argparse.ArgumentParser()

    argParser.add_argument("-w", "--wsis", help="The Path to the slides")
    argParser.add_argument("-t", "--tiles", help="the path where the tiles are supposed to be stored")
    argParser.add_argument("-j", "--jsons", help="the path to the folder containing the json files")
    argParser.add_argument("-s", "--size",type=int, default=500, help="tile size")
    argParser.add_argument("-l", "--level",type=int, default=0, help="magnification Level")

    args = argParser.parse_args()

    slides = args.wsis
    tiles = args.tiles
    jsons = args.jsons
    tileSize  = args.size
    level = args.level

    diags = ["MET", "MEL", "MEN"]

    tileKryos(slides, tiles, jsons,tileSize, level, diags)