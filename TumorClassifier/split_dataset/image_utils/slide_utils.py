def getPreparationFromFileName(fileName):
    prepString = fileName.split("-")[3]
    prepString = fileName.replace(".svs","")
    if not prepString[-1].isdigit():
        return prepString[-1]
    else:
        return prepString[-2]


def getDiagnosisFromFileName(fileName):
    diag = fileName.split("-")[0]
    if diag.startswith('A'):
        return 'A'
    elif diag.startswith('GBM'):
        return 'GBM'
    elif diag.startswith('O'):
        return 'O'


def multipleNNumber(fileName):
    prepString = fileName.split("-")[3]
    prepString = fileName.replace(".svs","")
    if not prepString[-1].isdigit():
        return False
        
    else:
        return True