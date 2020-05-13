import numpy

def normalize(cortList):
    normCortList = []
    cortLength = len(cortList[0])
    listLength = len(cortList)
    vertical = []
    for i in range(0, cortLength):
        vertical.append(__normList([cortList[j][i] for j in range(0, listLength)]))
    print(listLength)
    print(cortLength)
    print(len(vertical))
    print(len(vertical[0]))
    for i in range(0, listLength):
        cort = []
        for j in range(0, cortLength):
            cort.append(vertical[j][i])
        normCortList.append(tuple(cort))
    return normCortList



def __normList(valList):
    normL = []
    maxValue = max(valList)
    minValue = min(valList)
    for value in valList:
        if (maxValue - minValue == 0):
            normL.append(0)
        else:
            normL.append((value - minValue) / (maxValue - minValue))
    return normL


