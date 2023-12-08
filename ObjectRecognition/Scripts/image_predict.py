import sys
import model as core
import warnings

warnings.filterwarnings('ignore')

model_path = sys.argv[1]
image_path = sys.argv[2]

result = zip(*core.make_prediction(model_path, image_path))

for res in result:
  print(f"{(res[1] * 100):5.2f}%\t{res[0]}\n")

exit()