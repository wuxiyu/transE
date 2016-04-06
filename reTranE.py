from tranE import *

def loadData(str):
    fr = open(str)
    sArr = [line.strip().split("\t") for line in fr.readlines()]
    datArr = [[float(s) for s in line[1][1:-1].split(", ")] for line in sArr]
    nameArr = [line[0] for line in sArr]
    dic = {}
    for name, vec in zip(nameArr, datArr):
        dic[name] = vec
    return dic

if __name__ == '__main__':
    dirEntityVector = "c:\\entityVector.txt"
    entityList = loadData(dirEntityVector)
    dirRelationVector = "c:\\relationVector.txt"
    relationList = loadData(dirRelationVector)
    dirTrain = "C:\\data\\train.txt"
    tripleNum, tripleList = openTrain(dirTrain)
    transE = TransE(entityList, relationList, tripleList, learingRate = 0.001, dim = 30)
    transE.transE(100000)
    transE.writeRelationVector("c:\\relationVector.txt")
    transE.writeEntilyVector("c:\\entityVector.txt")