import os
import shutil


if __name__ == '__main__':

    inPath = ""
    outPath = ""

    slides = os.listdir("")
    for slide in slides:
        if slide.endswith(".jpg"):
            shutil.move(os.path.join(inPath,slide),os.path.join(outPath,slide))