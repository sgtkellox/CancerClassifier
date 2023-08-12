import os





if __name__ == '__main__':

    pathToTxtFiles = r"D:\TMA_Results"

    pathToRes = r"C:\Users\felix\Desktop\tma\result.txt"

    

    counter = 0

    with open(pathToRes, "a") as res:
    

        for output in os.listdir(pathToTxtFiles):
            print(output)
            filePath = os.path.join(pathToTxtFiles,output)
            res.write("---------------------------- \n")
            res.write(output+"\n")
            res.write("---------------------------- \n")
            with open(filePath) as f:
                lines = f.readlines()
                if counter > 0:
                    lines = lines[1:]
                f.close()
            for line in lines:
                res.write(line)
            counter +=1
    res.close()

