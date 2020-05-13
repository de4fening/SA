import util
import neuronLayer as NL

def parseTuple(line: str):
    result = []
    for val in line.split(";"):
        result.append(float(val))
    return tuple(result)

if __name__ == "__main__":
    skynet = NL.NeyronLayer(5, 16, 0.2, True, 0.8, 0.8, True)
    file = open("data.txt", "r")
    lines = file.readlines()
    file.close()
    education = []
    for line in lines:
        education.append(parseTuple(line))
    normal = util.normalize(education)
    for point in normal:
        skynet.addRes(point)
    skynet.showResults()

