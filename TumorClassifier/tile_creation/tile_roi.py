from genericpath import isfile
import json
from re import S
from openslide import OpenSlide
import os
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

import cv2
import numpy as np

import multiprocessing

from make_tiles import make_tiles

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
    
    

def tileAnnotatedArea(slidePath,geojson, tilePath, tileSize, level):
    
    slideName = getSlideName(slidePath)
    
    with open(geojson) as f:
        geojson_data = json.load(f)

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
    
    print(f"Slide w0 dimensions are {w0}x{h0}.", flush=True)
    print("Tiling...", flush=True)
    print("-----")
    
    for entry in geojson_data['features']:
        
        if entry['geometry']["type"] != "Polygon":
            continue
        coordinates = entry['geometry']['coordinates'][0]
        
        

        # Create a polygon from the coordinates using the shapely library
        roi_polygon = Polygon(coordinates)

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
                
                # Check if the tile intersects with the Region of Interest (ROI) polygon: add it to the list
                if percentage>0.5:
                   
                   safePath = os.path.join(tilePath,slideName+"_"+str(int(x/factor))+"_"+str(int(y/factor))+".jpg")
                   tileRGB = tile.convert('RGB')
           
                   tileNP = np.array(tileRGB)
                   tileNP = cv2.cvtColor(tileNP, cv2.COLOR_BGR2RGB)
                   cv2.imwrite(safePath,tileNP)
           

    
    slide.close()
    
def printInfo(subSet):
    print("started processing subset" +str(i)+ " of " + str(cpus)+"\n")
    print(" tiling subset with:")
    for slide in subSet:
        print("slideName " + slide)
    
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

def tileKryos(slidePath, tilePath, jsonPath,tileSize, level):
    for slide in os.listdir(slidePath):       
        split = slide.split(".")[0].split("-")              
        if split[3] =="K":            
            if geoJsonExists(slide,jsonPath):
                json = getJsonPath(slide,jsonPath)
                filePath = os.path.join(slidePath,slide)
                tileAnnotatedArea(filePath,json, tilePath, tileSize,level)
            elif not geoJsonExists(slide,jsonPath) and split[4]=="Q2":
                filePath = os.path.join(slidePath,slide)
                make_tiles(filePath, tilePath, tileSize, level)
                    
                    
                    

def tileKryosSubset(slidePath, tilePath, jsonPath,tileSize, level, subset):
     for slide in subset:       
        split = slide.split(".")[0].split("-")      
           
        if split[3] =="K":                       
            if geoJsonExists(slide,jsonPath):
                json = getJsonPath(slide,jsonPath)
                filePath = os.path.join(slidePath,slide)
                tileAnnotatedArea(filePath,json, tilePath, tileSize,level)
            else: 
                if split[4]=="Q2":
                    filePath = os.path.join(slidePath,slide)
                    make_tiles(filePath, tilePath, tileSize, level)
    
                
                 
             

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

    diags = ["A","EPN","EPNM","EPNS","GBM","H3","O", "PA", "PXA"]
    
    cpus = multiprocessing.cpu_count()-4

    print("Number of cpu : ", cpus)

    images = os.listdir(slides)
    filteredImages = [image for image in images if image.split("-")[0] in diags]

    numImages = len(filteredImages)

    procs = []
    cpus = int(cpus)
    
    


    for i in range(1,cpus+1):
        subSet = filteredImages[int((i-1)*numImages/cpus):int(i*numImages/cpus)]
        printInfo(subSet)
        tilingProc = multiprocessing.Process(target=tileKryosSubset,args=(slides, tiles, jsons,tileSize, level, subSet))
        procs.append(tilingProc)
        tilingProc.start()
        

    for process in procs:
        process.join()
    print("folder finished")
    


    