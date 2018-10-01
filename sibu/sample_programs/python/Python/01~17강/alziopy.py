from functools import *
def intersect(*ar):
    return reduce(__intersectSC, ar)

def __intersectSC(listX, listY):
    setList = []
    for x in listX:
        if x in listY:
            setList.append(x)
    
    return setList

def difference(*ar):
    setList = []
    inersectSet = intersect(*ar)
    unionSet = union(*ar)
    for x in unionSet:
        if not x in inersectSet:
            setList.append(x)
    return setList

def union(*ar):
    setList = []
    for item in ar:
        for x in item:
            if not x in setList:
                setList.append(x)
                
    return setList