from torch import nn, optim
from torch.optim import lr_scheduler
import model as core

dataloaders, label_class_dict = core.init_dataloaders()
model = core.model_create(len(label_class_dict))
model = model.to(core.device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
model = core.model_train(model, dataloaders, criterion, optimizer, scheduler, epoch_count=18)

core.create_checkpoint(model, label_class_dict, criterion, optimizer, scheduler)

exit()