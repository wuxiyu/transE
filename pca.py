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

def getEig(inputM):
    covM = cov(inputM, rowvar=0)
    s,V = linalg.eig(covM)
    return s,V

def judge(s):
    s.sort()
    s = s[::-1] 
    bili = []
    i = 0
    sum1 = 0.0
    sum2 = s[0]
    while i<len(s)-1:
        sum1=sum1+s[i]
        sum2=sum2+s[i+1]
        bili.append(sum1/sum2)
        i+=1
    plt.plot(range(len(bili)), bili, 'b*')
    plt.plot(range(len(bili)), bili, 'r')
    for xy in zip(range(len(bili)),bili):
        plt.annotate(xy[1], xy=xy, xytext=(-20, 10), textcoords = 'offset points')
    plt.xlabel("eigenvector")
    plt.ylabel("eigenvalue")
    plt.title('fangchabili')
    plt.legend()
    plt.show()
    return bili

def getbaifenbi(bili, num):
    i = 1
    for b in bili:
        if b > num:
            break
        i+=1
    return i

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
        #,textcoords = 'offset points',bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        # #arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

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
    #s,V = getEig(mat)
    #bili= judge(s)
    #k = getbaifenbi(bili, 0.9)
    k = 2
    a, b = pca(mat, k)
    plotV(a, nameEntity)
