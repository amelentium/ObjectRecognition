import sys
import model as core
import warnings

warnings.filterwarnings("ignore")

modelPath = sys.argv[1]
imagePath = sys.argv[2]

result = zip(*core.make_prediction(modelPath, imagePath))

for res in result:
  print(str(res[0]) + '\t' + "{:5.2f}".format(res[1] * 100) + '%\n')

exit()