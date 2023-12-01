import sys
import model as core

modelPath = sys.argv[1]
imagePath = sys.argv[2]
resultPath = sys.argv[3]

result = zip(*core.make_prediction(modelPath, imagePath))

f = open(resultPath, 'w')

for res in result:
  f.write(str(res[0]) + '\t' + str(res[1]) + '\n')

f.close()

exit()