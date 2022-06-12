import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
import sys
import model as core

#Image preparation
imageName = sys.argv[1]
image = Image.open(f'../Images/user_images/{imageName}.jpg')
image = TF.resize(image, 256)

upper_pixel = (image.height - 224) // 2
left_pixel = (image.width - 224) // 2
image = TF.crop(image, upper_pixel, left_pixel, 224, 224)

image = TF.to_tensor(image)
image = TF.normalize(image, core.mean, core.std)

#Model load
model, _, _, _ = core.load_checkpoint()
model = model.to(core.device)

#Make predict
probs, classes = core.predict(model, image)
probs = probs.data.cpu()
probs = probs.numpy().squeeze()

#Create and save image
classes = classes.data.cpu()
classes = classes.numpy().squeeze()
classes = [core.cat_label_to_name[clazz].title() for clazz in classes]

fig = plt.figure(figsize=(4, 10))
ax1 = fig.add_subplot(2, 1, 1, xticks=[], yticks=[])
core.image_show(image, ax1)
ax2 = fig.add_subplot(2, 1, 2, xticks=[], yticks=[])
ax2.barh(np.arange(5), probs)
ax2.set_yticks(np.arange(5))
ax2.set_yticklabels(classes)
ax2.set_ylim(-1, 5)
ax2.invert_yaxis()
ax2.set_xlim(0, 1)
ax2.set_title('Class Probability')
    
for i in range(1, 6):
    ax2.text(0.4234, (i-1 + i*0.1 - (i-1)*0.1), f'{probs[i-1]:.4f}')

plt.savefig(f'../Images/user_images/{imageName}_result.jpg', bbox_inches='tight')
plt.close()
exit()