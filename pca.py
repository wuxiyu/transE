from numpy import *
import matplotlib.pyplot as plt
import pylab

def loadData(str):
    fr = open(str)
    sArr = [line.strip().split("\t") for line in fr.readlines()]
    datArr = [[float(s) for s in line[1][1:-1].split(", ")] for line in sArr]
    matA = mat(datArr)
    print(matA.shape)
    nameArr = [line[0] for line in sArr]
    return matA, nameArr

def pca(inputM, k):
    covM = cov(inputM, rowvar=0)
    s, V = linalg.eig(covM)
    paixu = argsort(s)
    paixuk = paixu[:-(k+1):-1]
    kwei = V[:,paixuk]
    outputM = inputM * kwei
    chonggou = (outputM * kwei.T)
    return outputM,chonggou

def plotV(a, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    print("aaa")
    font = { 'fontname':'Tahoma', 'fontsize':0.5, 'verticalalignment': 'top', 'horizontalalignment':'center' }
    ax.scatter(a[:,0], a[:,1], marker = ' ')
    ax.set_xlim(-0.8,0.8)
    ax.set_ylim(-0.8,0.8)
    i = 0
    for label, x, y in zip(labels, a[:, 0], a[:, 1]):
        i += 1
        s = random.uniform(0,100)
        if i<14951:
            if s > 3.1:
                continue
        else:
            if s > 6.7:
                continue
        ax.annotate(label, xy = (x, y), xytext = None, ha = 'right', va = 'bottom', **font)


    plt.title('TransE pca2dim')
    plt.xlabel('X')
    plt.ylabel('Y')
    print("ddd")
    plt.savefig('plot_with_labels', dpi = 3000, bbox_inches = 'tight' ,orientation = 'landscape', papertype = 'a0')
if __name__ == '__main__':
    dirEntity = "c:\\entityVector.txt"
    dirRelation = "c:\\relationVector.txt"
    matEntity, nameEntity = loadData(dirEntity)
    matRelation, nameRelation = loadData(dirRelation)
    mat = row_stack((matEntity, matRelation))
    print(mat.shape)
    nameEntity.extend(nameRelation)
    k = 2
    a, b = pca(mat, k)
    plotV(a, nameEntity)
