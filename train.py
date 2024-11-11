import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets, models
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR-10 Classification with PyTorch Lightning')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training and validation')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--milestones', type=int, nargs='+', default=[60, 80], help='Milestones for LR scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR scheduler gamma')
    parser.add_argument('--project', type=str, default='CriticalPeriodCifar10', help='Wandb project name')
    parser.add_argument('--log_dir', type=str, default='logs/', help='Directory for CSV logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/', help='Directory for model checkpoints')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loaders')
    return parser.parse_args()

args = parse_args()

# Data augmentation: random translations up to 4 pixels and random horizontal flipping
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(4/32, 4/32)),  # Translate up to 4 pixels
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # CIFAR-10 mean and std
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)
val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

# Function to modify ResNet18
def create_modified_resnet18():
    # Load ResNet18 without pretraining
    model = models.resnet18()
    # Modify the first convolutional layer to accept 32x32 images
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Remove the max pooling layer to retain more spatial information
    model.maxpool = nn.Identity()
    # Adjust the final fully connected layer to output 10 classes for CIFAR-10
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    return model

# Define the LightningModule
class CIFAR10Classifier(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = create_modified_resnet18()
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        acc = (outputs.argmax(dim=1) == targets).float().mean()
        # Log training loss and accuracy
        self.log('train_loss', loss)
        self.log('train_acc', acc, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        acc = (outputs.argmax(dim=1) == targets).float().mean()
        # Log validation loss and accuracy
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)
        return [optimizer], [scheduler]

# Initialize loggers
wandb_logger = WandbLogger(project=args.project)
csv_logger = CSVLogger(save_dir=args.log_dir, name='cifar10')

# Model checkpointing
checkpoint_callback = ModelCheckpoint(
    monitor='val_acc',
    dirpath=args.checkpoint_dir,
    filename='cifar10-{epoch:02d}-{val_acc:.2f}',
    save_top_k=3,
    mode='max',
)

# Initialize the Trainer
trainer = pl.Trainer(
    max_epochs=args.max_epochs,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1 if torch.cuda.is_available() else None,
    logger=[wandb_logger, csv_logger],
    callbacks=[checkpoint_callback]
)

# Initialize the model
model = CIFAR10Classifier()

# Start training
trainer.fit(model, train_loader, val_loader)