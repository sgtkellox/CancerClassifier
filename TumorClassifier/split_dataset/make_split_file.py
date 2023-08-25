
import os


def documentSplit(path,docPath):
    split = os.listdir(path)
    outpath = os.path.join(docPath,"doc.txt")

    for part in split:
        partPath = os.path.join(path,part)
        with open(outpath, 'a') as doc:
            print(part)
            doc.write(part)
            doc.write('\n')
            doc.write('------------------------')
            doc.write('\n')
            for folder_path, folders, slides in os.walk(partPath):
                for slide in slides:
                    doc.write(slide)
                    doc.write('\n')
            doc.write('------------------------')
            doc.write('\n')

