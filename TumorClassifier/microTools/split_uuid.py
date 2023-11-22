

def extractNNumbersFromFile(path):

    with open(path,"r") as f:
        lines = f.readlines()
    lines = [x.rstrip() for x in lines]
    lines = [x.split("/")[5] for x in lines]
    lines = [x.split(".")[0] for x in lines]
    
    
    f.close()
    
    print(lines)

    

    return lines


lines = extractNNumbersFromFile(r"D:\slides\allreadyScanned\notFound.txt")

for line in lines:
    with open(r"D:\slides\allreadyScanned\uuids.txt","a") as copiedFile:
                copiedFile.write(line)
                copiedFile.write('\n')