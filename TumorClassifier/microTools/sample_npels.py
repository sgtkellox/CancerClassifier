import os 
import shutil

def extractNNumberFromWsi(wsi):

    wsi = wsi.split(".")[0]
    split = wsi.split("-")

    if  wsi.startswith("Inf"):
        
        nNumber = split[2]+"-"+split[3]

    elif wsi.startswith("G-"):

        if split[1] == "Inf":
            nNumber = split[3]+"-"+split[4]
        else:
            nNumber = split[2]+"-"+split[3]

    elif wsi.startswith("WSI"):
        
        nNumber = split[2]+"-"+split[3]
    else:
       
        nNumber = split[1]+"-"+split[2]
    return nNumber

def collectNNumbers(path):

    nNumbers = []

    for slide in os.listdir(path):
        nNumber = extractNNumberFromWsi(slide)

        nNumbers.append(nNumber)

    return nNumbers


def diff(li1, li2):
    diffList = [i for i in li1 + li2 if i not in li1 or i not in li2]
    return diffList


def inANotB(list1,list2):
    diff = []
    for entry in list1:
        if not entry in list2:
            diff.append(entry)
    return diff


def inBoth(list1,list2):

    both = []
    for entry in list1:
        if entry in list2:
            both.append(entry)
    return both


def copyToFolder(numbers,src, dest):

    slides = os.listdir(src)

    for nNumber in numbers:
        for slide in slides:
            if nNumber in slide:
                slidepath = os.path.join(src,slide)
                slide = "G-"+slide
                destpath = os.path.join(dest,slide)
                shutil.move(slidepath,destpath)

def rename(path):
    for file in os.listdir(path):
        newName = file.replace("Inf-","")

        oldName = os.path.join(path,file)

        newFullName = os.path.join(path,newName)

        os.rename(oldName,newFullName)


def multipleSlides(path):
    for slide in os.listdir(path):
        print()
    
   



if __name__ == '__main__':

    kPath1 = r"E:\slides\kryoQ2"
    kPath2 = r"E:\slides\kryoQ1"
    kPath3 = r"E:\slides\kryoQ0"
    kPath4 = r"E:\slides\Inf_kryo"
    kPath5 = r"E:\slides\WSI_Kryo"






    qPath1 = r"E:\slides\smear"
    #qPath2 = r"E:\slides\smearQ0"
    qPath3 = r"E:\slides\Inf_Smear"

   





    kNumbers1 = collectNNumbers(kPath1)
    kNumbers2 = collectNNumbers(kPath2)
    kNumbers3 = collectNNumbers(kPath3)
    kNumbers4 = collectNNumbers(kPath4)
    kNumbers5 = collectNNumbers(kPath5)

   



    qNumbers1 = collectNNumbers(qPath1)
    #qNumbers2 = collectNNumbers(qPath2)
    qNumbers3 = collectNNumbers(qPath3)

   




    kNumbers = kNumbers4 

    qNumbers = qNumbers3


    #print(len(nNumbers2))


    alsoSolid = inBoth(qNumbers3,kNumbers1)

    

    alsoInf = inBoth(qNumbers3,kNumbers4)

    

    #res =  list(set(diff)-set(alsoSolid))

    infAndSolid = set(alsoSolid+alsoInf)

    print(len(infAndSolid))

    for entry in infAndSolid:
        print(entry)

    copyToFolder(infAndSolid,r"E:\slides\Inf_Smear",r"E:\slides\InfAndSolid")








    

        


