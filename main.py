import numpy as np
import ast
# ------- Reading the files needed -------
fileIn = "HW1/data/1.in"
fileJson = "HW1/data/1.json"
fileOut = "HW1/data/1.out"

with open(fileIn, "r") as fIn:
    list = [[float(i) for i in line.split(' ')] for line in fIn]
fIn.close()

with open(fileJson, "r") as fJson:
    var = fJson.read()
    dictJson = ast.literal_eval(var)
fJson.close()

learningRate = dictJson['learning rate']
numIter = dictJson['num iter']

# ------- Creating array with starting 1's -------
for i in list:
    i.insert(0,1)

# ------- start of calculating wStar -------

# Only used to get the y-values
arr = np.array(list)
arrTrans = arr.transpose()
yVals = arrTrans[-1]
#print(yVals)

#removes the last column
for i in list:
    i.pop(-1)

#creating matrix phi
phi = np.array(list)
#print(phi)
#print(phi[0])

phiTrans = phi.transpose()
#print(phiTrans)

#result is the multiplication of phi transpose and phi
result = np.dot(phiTrans, phi)
#print(result)

phiInvert = np.linalg.inv(result)
#print(arrInvert)

wStar = np.dot(phiInvert,np.dot(phiTrans, yVals))
#print(wStar)

# ------- Gradiant decent calculations -------
wInital = 0

#Makes the yVals into a column matrix
yVals = yVals[...,None]
print(yVals)
tempvar = phiInvert.transpose()

for i in range(numIter):
    gradDecent = learningRate * (yVals - np.dot(phiInvert,phi) * phi[i])
    wStarInital = wInital - gradDecent
print(wInital)

with open(fileOut, "w") as fOut:
    fOut.write(wStar)
fOut.close()