import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sys
import model as core

#Model load
model, _, _, _, _ = core.load_checkpoint()
model = model.to(core.device)

#Image preparation
imageName = sys.argv[1]
imagePath = f'../Images/user_images/{imageName}.jpg'
image, image_dn = core.image_process(imagePath)

#Make predict
probs, classes = core.predict(model, image)
probs = probs.data.cpu()
probs = probs.numpy().squeeze()

probs_dn, classes_dn = core.predict(model, image_dn)
probs_dn = probs_dn.data.cpu()
probs_dn = probs_dn.numpy().squeeze()

if (probs_dn[0] > probs[0]):
    probs = probs_dn
    classes = classes_dn

#Create and save image
classes = classes.data.cpu()
classes = classes.numpy().squeeze()
classes = [core.label_class_dict[_class_].title() for _class_ in classes]

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