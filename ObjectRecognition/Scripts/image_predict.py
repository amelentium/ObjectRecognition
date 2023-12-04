import sys
import model as core
import warnings

warnings.filterwarnings('ignore')

model_path = sys.argv[1]
image_path = sys.argv[2]

result = zip(*core.make_prediction(model_path, image_path))

for res in result:
  print(str(res[0]) + '\t' + "{:5.2f}".format(res[1] * 100) + '%\n')

exit()