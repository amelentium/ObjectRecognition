import sys
import model as core
import warnings

warnings.filterwarnings('ignore')

model_path = sys.argv[1]
train_images_path = sys.argv[2]
val_images_path = sys.argv[3]

core.train(model_path, train_images_path, val_images_path)

exit()