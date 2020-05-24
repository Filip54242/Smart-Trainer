import numpy


def checkMesage(message):
    return len(message) % 7 == 0


def makeParity(message):
    matrix = numpy.asarray(list(message), dtype=int)
    rows, colons = (len(matrix) // 7), 7
    colonResult = ' '
    for indexFirst in range(colons):
        count = 0
        for indexSecond in range(rows):
            if matrix[indexFirst + indexSecond * 7]:
                count += 1
        colonResult += str(count % 2) + " "
    matrix = matrix.reshape((rows, colons))
    for row in matrix:
        count = 0
        for element in row:
            if element == 1:
                count += 1
                print(str(row) + "---> " + str(count % 2))
    print(colonResult)


message = input("Type the message :")
if checkMesage(message):
    makeParity(message)
else:
    print("Invalid message!")
