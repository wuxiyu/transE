from random import uniform, sample
from numpy import *

class TransE:
    def __init__(self, entityList, relationList, tripleList, margin = 1, learingRate = 0.01, dim = 10, lambd = 0.4):
        self.margin=margin
        self.learingRate = learingRate
        self.dim = dim#向量维度
        self.entityList = entityList#一开始，entityList是entity的list；初始化后，变为字典，key是entity，values是其向量。
        self.relationList = relationList#理由同上
        self.tripleList = tripleList#理由同上
        self.lambd = lambd
        self.loss = 0

    def initialize(self):
        '''
        初始化向量
        '''
        entityVectorList = {}
        relationVectorList = {}
        for entity in self.entityList:
            n = 0
            entityVector = []
            while n < self.dim:
                ram = init(self.dim)#初始化的范围
                entityVector.append(ram)
                n += 1
            entityVector = norm(entityVector)#归一化
            entityVectorList[entity] = entityVector
        print("entityVector初始化完成，数量是%d"%len(entityVectorList))
        for relation in self. relationList:
            n = 0
            relationVector = []
            while n < self.dim:
                ram = init(self.dim)#初始化的范围
                relationVector.append(ram)
                n += 1
            relationVector = norm(relationVector)#归一化
            relationVectorList[relation] = relationVector
        print("relationVectorList初始化完成，数量是%d"%len(relationVectorList))
        self.entityList = entityVectorList
        self.relationList = relationVectorList

    def transE(self, cI = 20):
        print("训练开始")
        for cycleIndex in range(cI):
            if cycleIndex%10000==0:
                print("第%d次循环"%cycleIndex)
                print(self.loss)
                self.loss = 0
                self.writeRelationVector("c:\\relationVector.txt")
                self.writeEntilyVector("c:\\entityVector.txt")
            Sbatch = self.getSample()
            Tbatch = []#元组对（原三元组，打碎的三元组）的列表 ：{((h,r,t),(h',r,t'))}
            for sbatch in Sbatch:
                tripletWithCorruptedTriplet = (sbatch, self.getCorruptedTriplet(sbatch))
                if(tripletWithCorruptedTriplet not in Tbatch):
                    Tbatch.append(tripletWithCorruptedTriplet)
            self.update(Tbatch)

    def getSample(self, size = 500):
        return sample(self.tripleList, size)

    def getCorruptedTriplet(self, triplet):
        '''
        training triplets with either the head or tail replaced by a random entity (but not both at the same time)
        :param triplet:
        :return corruptedTriplet:
        '''
        i = uniform(-1, 1)
        if i < 0:#小于0，打坏三元组的第一项
            while True:
                entityTemp = sample(self.entityList.keys(), 1)[0]
                if entityTemp != triplet[0]:
                    break
            corruptedTriplet = (entityTemp,triplet[1], triplet[2])
        else:#大于等于0，打坏三元组的第二项
            while True:
                entityTemp = sample(self.entityList.keys(), 1)[0]
                if entityTemp != triplet[1]:
                    break
            corruptedTriplet = (triplet[0],entityTemp, triplet[2])
        return corruptedTriplet

    def update(self, Tbatch):
        i = 0
        while i < len(Tbatch):
            tripletWithCorruptedTriplet = Tbatch[i]
            headEntityVector = array(self.entityList[tripletWithCorruptedTriplet[0][0]])#tripletWithCorruptedTriplet是原三元组和打碎的三元组的元组tuple
            tailEntityVector = array(self.entityList[tripletWithCorruptedTriplet[0][1]])
            relationVector = array(self.relationList[tripletWithCorruptedTriplet[0][2]])
            headEntityVectorWithCorruptedTriplet = array(self.entityList[tripletWithCorruptedTriplet[1][0]])
            tailEntityVectorWithCorruptedTriplet = array(self.entityList[tripletWithCorruptedTriplet[1][1]])

            distTriplet = distance(headEntityVector,tailEntityVector , relationVector)
            distCorruptedTriplet = distance(headEntityVectorWithCorruptedTriplet,tailEntityVectorWithCorruptedTriplet ,  relationVector)
            eg = self.margin + distTriplet - distCorruptedTriplet
            if eg > 0: #[function]+ 是一个取正值的函数
                self.loss += eg
                tempPositive = 2 * self.learingRate * (tailEntityVector - headEntityVector - relationVector)
                tempNegtative = 2 * self.learingRate * (tailEntityVectorWithCorruptedTriplet - headEntityVectorWithCorruptedTriplet - relationVector)

                temp1 = headEntityVector + tempPositive
                temp2 = tailEntityVector - tempPositive
                temp3 = relationVector + tempPositive - tempNegtative
                temp4 = headEntityVectorWithCorruptedTriplet - tempNegtative
                temp5 = tailEntityVectorWithCorruptedTriplet + tempNegtative

                headEntityVector = temp1
                tailEntityVector = temp2
                relationVector = temp3
                headEntityVectorWithCorruptedTriplet = temp4
                tailEntityVectorWithCorruptedTriplet = temp5

                #只归一化这几个刚更新的向量，而不是按原论文那些一口气全更新了
                self.entityList[tripletWithCorruptedTriplet[0][0]] = norm(headEntityVector.tolist())
                self.entityList[tripletWithCorruptedTriplet[0][1]] = norm(tailEntityVector.tolist())
                self.relationList[tripletWithCorruptedTriplet[0][2]] = norm(relationVector.tolist())
                self.entityList[tripletWithCorruptedTriplet[1][0]] = norm(headEntityVectorWithCorruptedTriplet.tolist())
                self.entityList[tripletWithCorruptedTriplet[1][1]] = norm(tailEntityVectorWithCorruptedTriplet.tolist())
            i += 1

    def writeEntilyVector(self, dir):
        print("写入实体")
        entityVectorFile = open(dir, 'w')
        for entity in self.entityList.keys():
            entityVectorFile.write(entity+"\t")
            entityVectorFile.write(str(self.entityList[entity]))
            entityVectorFile.write("\n")
        entityVectorFile.close()

    def writeRelationVector(self, dir):
        print("写入关系")
        relationVectorFile = open(dir, 'w')
        for relation in self.relationList.keys():
            relationVectorFile.write(relation + "\t")
            relationVectorFile.write(str(self.relationList[relation]))
            relationVectorFile.write("\n")
        relationVectorFile.close()

def init(dim):
    return uniform(-6/(dim**0.5),6/(dim**0.5))

def distance(h, t, r):
    s = h + r - t
    narray = array(s)
    narray2 = narray*narray
    sum = narray2.sum()
    return sum

def norm(list):
    '''
    归一化
    :param 向量:
    :return: 向量的平方和的开方后的向量
    '''
    var = linalg.norm(list)
    i = 0
    while i < len(list):
        list[i] = list[i]/var
        i += 1
    return list

def openDetailsAndId(dir,sp="\t"):
    idNum = 0
    list = []
    with open(dir) as file:
        lines = file.readlines()
        for line in lines:
            DetailsAndId = line.strip().split(sp)
            list.append(DetailsAndId[0])
            idNum += 1
    return idNum, list

def openTrain(dir,sp="\t"):
    num = 0
    list = []
    with open(dir) as file:
        lines = file.readlines()
        for line in lines:
            triple = line.strip().split(sp)
            if(len(triple)<3):
                continue
            list.append(tuple(triple))
            num += 1
    return num, list

if __name__ == '__main__':
    dirEntity = "C:\\data\\entity2id.txt"
    entityIdNum, entityList = openDetailsAndId(dirEntity)
    dirRelation = "C:\\data\\relation2id.txt"
    relationIdNum, relationList = openDetailsAndId(dirRelation)
    dirTrain = "C:\\data\\train.txt"
    tripleNum, tripleList = openTrain(dirTrain)
    print("打开TransE")
    transE = TransE(entityList,relationList,tripleList,dim = 30)
    print("TranE初始化")
    transE.initialize()
    transE.transE(300000)
    transE.writeRelationVector("c:\\relationVector.txt")
    transE.writeEntilyVector("c:\\entityVector.txt")
