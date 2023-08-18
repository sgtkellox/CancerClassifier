import os


def extractNNumberFromWsi(wsi):
    split = wsi.split("-")
    if wsi.lower().startswith("inf"):
        nNumber = split[2]+"-"+split[3]+"-"+split[4]
    elif wsi.lower().startswith("tumor"):
        nNumber = split[3]+"-"+split[4]+"-"+split[5]
    else:
        nNumber = "shit"
    return nNumber


if __name__ == '__main__':

    slidePath = r"E:\slides\Inf_Kryo"

    resPath = r"E:\slides\inf2.txt"

    with open(resPath, 'a') as f:
        for slide in os.listdir(slidePath):
            nNumber = extractNNumberFromWsi(slide)
            print(nNumber)
            f.write(nNumber)
            f.write("\n")
        f.close()
        
            