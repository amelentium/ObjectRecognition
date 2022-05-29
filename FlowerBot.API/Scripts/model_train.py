from torch import nn, optim
from torch.optim import lr_scheduler
import model as core

model = core.model_create()

model = model.to(core.device)
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
model = core.train_model(model, criterion, optimizer, scheduler, epoch_count=10)

core.create_checkpoint(model, criterion, optimizer, scheduler)
exit()