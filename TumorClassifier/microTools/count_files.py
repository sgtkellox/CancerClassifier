import os
import argparse

def countFiles(path):
    files = os.listdir(path)
    count = sum(map(lambda x : x.endswith(".jpg"), files))
    print(count)

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()

    
    argParser.add_argument("-p", "--path", help="Folder containing the files to be counted")

    args = argParser.parse_args()

    path = args.path

    countFiles(path)