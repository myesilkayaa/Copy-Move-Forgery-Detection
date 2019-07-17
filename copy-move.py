# -*- coding: utf-8 -*-

from PIL import Image
import numpy
import math
from scipy.fftpack import dct
from tqdm import tqdm
image = Image.open("forged1.png")
imageMask = Image.open("forged1_maske.png")

def griyeCevir(image):
    
    imageWidth,imageHeight = image.size
    np_im = numpy.array(image,dtype=numpy.int64)
    imageArray =numpy.zeros((imageHeight,imageWidth),dtype=numpy.float64)
    
    for i in range(imageHeight):
        for j in range(imageWidth):
            grayValue = int(0.299*np_im[i][j][0] + 0.587*np_im[i][j][1] + 0.114*np_im[i][j][2])
            imageArray[i][j]= grayValue
            
    grayImage = Image.fromarray(imageArray.astype('uint8'))    
    grayImage.show()
    print("Görüntü griye çevrildi.")
#    print(imageArray)
    return imageArray

def dct2(block):
    
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def initializing(image):
    
    blok_size = 8
    Q8Matrix = numpy.zeros((blok_size, blok_size),dtype = numpy.float64)
    
#    2 FARKLI KUANTALAMA MATRİSİ VARDIR. 2. MATRİS DAHA İYİ SONUC VERİYOR
    
#    Q8=[[16,11,10,16,24,40,51,61],
#    [12,12,14,19,26,58,60,55],
#    [14,13,16,24,40,57,69,56],
#    [14,17,22,29,51,87,80,62],
#    [18,22,37,56,68,109,103,77],
#    [24,35,55,64,81,104,113,92],
#    [49,64,78,87,103,121,120,101],
#    [72,92,95,98,112,100,103,99] ];
    
    Q8=[[17,	18,	24,	47,	99,	99,	99,	99],
        [18,	21,	26,	66,	99,	99,	99,	99],
        [24,	26,	56,	99,	99,	99,	99,	99],
        [47,	66,	99,	99,	99,	99,	99,	99],   
        [99,	99,	99,	99,	99,	99,	99,	99],
        [99,	99,	99,	99,	99,	99,	99,	99],
        [99,	99,	99,	99,	99,	99,	99,	99],
        [99,	99,	99,	99,	99,	99,	99,	99]]

        
    for i in range(blok_size):
        for j in range(blok_size):
            if(i==0 and j ==0):
                Q8Matrix[i][j] = 2*Q8[i][j]
            else:    
                Q8Matrix[i][j] = 2.5*Q8[i][j]
                
    imageArray = griyeCevir(image)
    size_y = imageArray.shape[0]
#    print(size_y)
    size_x = imageArray.shape[1]
    M = size_y - blok_size + 1
    N = size_x - blok_size + 1
    global sVector
    sVector = numpy.zeros((M*N,16),dtype = numpy.int16)
    blokNumber=0
    ofs = int(blok_size/2)     
    print("Görüntüye DCT uygulanıyor.")
    
    for i in tqdm(range (ofs, size_y-ofs + 1)):
        for j in range (ofs,size_x-ofs + 1):
            blokArray = numpy.zeros((blok_size,blok_size))
            blokNumber = blokNumber + 1
            for k in range(blok_size*blok_size):
                im_val = imageArray[int(i + (k / blok_size)-ofs)][int(j + (k % blok_size)-ofs)]
                blokArray[int(k / blok_size)][int(k % blok_size)] = im_val
            result = numpy.round(dct2(blokArray)/Q8Matrix)          
            zzresult = zigzagScanning(result)
            for y in range(16):
                sVector[blokNumber-1][y] = zzresult[y]
                
    print("Uzaklıklar hesaplanmaya başladı")
    matching(sVector)
#matchedBlocks = numpy.zeros(37249,dtype = numpy.float64) 


def zigzagScanning(blockArray):
    
    zigzagVector = numpy.zeros(16,dtype=numpy.float64)
    zigzagVector[0]  = blockArray[0][0]
    zigzagVector[1]  = blockArray[0][1]
    zigzagVector[2]  = blockArray[1][0]
    zigzagVector[3]  = blockArray[2][0]
    zigzagVector[4]  = blockArray[1][1]
    zigzagVector[5]  = blockArray[0][2]
    zigzagVector[6]  = blockArray[0][3]
    zigzagVector[7]  = blockArray[1][2]
    zigzagVector[8]  = blockArray[2][1]
    zigzagVector[9]  = blockArray[3][0]
    zigzagVector[10] = blockArray[4][0]
    zigzagVector[11] = blockArray[3][1]
    zigzagVector[12] = blockArray[2][2]
    zigzagVector[13] = blockArray[1][3]
    zigzagVector[14] = blockArray[0][4]
    zigzagVector[15] = blockArray[0][5]

    return zigzagVector

def matching(sVector):
#1
    
    sizew,sizeh = image.size
    M = sizew - 7
    np_im = numpy.array(image,dtype=numpy.int64)
    np_out =numpy.zeros((sizeh,sizew),dtype=numpy.float64)
    size_y = sVector.shape[0]

#    global ind
    ind = numpy.lexsort((sVector[:,15],sVector[:,14],sVector[:,13],sVector[:,12],sVector[:,11],sVector[:,10],sVector[:,9],
    sVector[:,8],sVector[:,7],sVector[:,6],sVector[:,5],sVector[:,4],sVector[:,3],sVector[:,2],sVector[:,1] ,sVector[:,0]   ))
    global sortedVector
    sortedVector = sVector[ind]
    global matchIndexes
    matchIndexes=[]
    global shiftMatrix
    shiftMatrix=[[0,0,0]]
    
    for i in tqdm(range(size_y-1)):
        shift1 = 0
        shift2 = 0
        if(numpy.array_equal(sortedVector[i],sortedVector[i+1])):
            a1 = int(ind[i]/M)
            a2 = ind[i]%M
            b1 = int(ind[i+1]/M)
            b2 = ind[i+1]%M
            shift1 = a1-b1
            shift2 = a2-b2
            distance = int(math.sqrt((a1-b1)**2+(a2-b2)**2))
            if( distance < 16):
                continue
        else:
              continue

        kontrol = 0
        for j in range(len(shiftMatrix)):  
            if(shiftMatrix[j][0]==shift1 and shiftMatrix[j][1]==shift2):
                indx = shiftMatrix.index([shiftMatrix[j][0],shiftMatrix[j][1],shiftMatrix[j][2]])
                shiftMatrix[indx][2] = shiftMatrix[indx][2] + 1
                matchIndexes.append([ind[i],ind[i+1],shift1,shift2])
                kontrol = 1
                break
        if(kontrol == 0):
            shiftMatrix.append([shift1,shift2,1])
            matchIndexes.append([ind[i],ind[i+1],shift1,shift2])

    maxV = 0
    shift1 = 0
    shift2 = 0
    global match
    match = []
    
    for i in range(len(shiftMatrix)):
        if(shiftMatrix[i][2]>maxV):
            maxV = shiftMatrix[i][2]
            shift1 = shiftMatrix[i][0]
            shift2 = shiftMatrix[i][1]
            
#    print("shiftX :",+shift1)
#    print("shiftY :",+shift2)        
    
    for i in range(len(matchIndexes)):
        if(matchIndexes[i][2]==shift1 and matchIndexes[i][3]==shift2):
            match.append([matchIndexes[i][0],matchIndexes[i][1]])
    B = 8
    for i in match:
        
        h0=int(i[0]/(sizew-B+1))
        w0=int(i[0]%(sizew-B+1))
        
        for x in range(h0,h0+B):
            for y in range(w0,w0+B):
                np_im[x][y][0] = 0
                np_im[x][y][1] = 0
                np_im[x][y][2] = 255
                
                np_out[x][y] = 255
               
#        np_im[h0:h0+B,w0:w0+B]=blueBlock
        h1=int(i[1]/(sizew-B+1))
        w1=int(i[1]%(sizew-B+1))
        
        for x in range(h1,h1+B):
            for y in range(w1,w1+B):
                np_im[x][y][0] = 0
                np_im[x][y][1] = 0
                np_im[x][y][2] = 255
                
                np_out[x][y] = 255
                
    img = Image.fromarray(np_im.astype('uint8'))    
    out = Image.fromarray(np_out.astype('uint8'))    
    img.show()
    out.show()     
#    img.save("2_gj90.png")
#    out.save("2_gj90_mask.png")

    accuracy(np_out,imageMask)

def accuracy(imgOut,imgDest):
    
    sizew,sizeh = imgDest.size
    imgDest = numpy.array(imgDest,dtype=numpy.int64)
    
    DP = 0
    YP = 0
    YN = 0
    for i in range(sizeh):
        for j in range(sizew):
        
            if(imgDest[i][j][0]==255 and imgOut[i][j]==255):
                
                DP = DP + 1
            
            elif(imgDest[i][j][0]==0 and imgOut[i][j]==255):
                
                YP = YP + 1
            
            elif(imgDest[i][j][0]==255 and imgOut[i][j]==0):
                
                YN = YN + 1
    
    precision = DP/(DP+YP)
    recall = DP/(DP+YN)
    
    F1 = 2*(precision*recall)/(precision+recall)
    
    print("Accuracy : ", + F1)

#DCT TRANSFORM-İŞLEM SÜRESİ UZUN SÜRÜYOR.
def c(x):
    
    if(x==0):
        return 1/math.sqrt(2)
    else:
        return 1
    
def dctTransform(blockArray):
    
    size = len(blockArray)
    Q8Matrix = numpy.zeros((size, size),dtype = numpy.float64)
    Q8=[[16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99] ];
    
    for i in range(size):
        for j in range(size):
            if(i==0 and j ==0):
                Q8Matrix[i][j] = 2*Q8[i][j]
            else:    
                Q8Matrix[i][j] = 2.5*Q8[i][j]
  
    dctArray = numpy.zeros((size,size))
    for i in range(size):
        for j in range(size):
            araDeger = (1/(math.sqrt(2*size)))*c(i)*c(j)
            sonuc = 0
            for x in range(size):
                for y in range(size):
                    sonuc = blockArray[x][y]*math.cos(((2*x+1)*i*math.pi)/(2*size))*math.cos(((2*y+1)*j*math.pi)/(2*size)) + sonuc
            dctArray[i][j] = round((araDeger * sonuc)/Q8Matrix[i][j])
            araDeger = 0

    return dctArray                   
                
