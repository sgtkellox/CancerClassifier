


def mergeGaps(gaps, tileSize, k):
    
    for index, gap in enumerate(gaps.copy()[0:-1]):               
        if gaps[index+1][0] - gap[1] <= k *tileSize:
            mergedGap = (gap[0], gaps[index+1][1])
            del gaps[index+1]
            del gaps[index]          
            gaps.insert(index,mergedGap)
            mergeGaps(gaps,tileSize,k)
            break

    return gaps



gaps = [(22528, 25088), (26112, 28672), (28672, 30208), (30208, 35328), (36352, 40448), (40448, 47104), (66560, 76800), (78336, 83456), (102400, 119296)]     

 
gaps = mergeGaps(gaps,512,4)

print(gaps)
        