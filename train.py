import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn.functional as F
import random

def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR-10 Classification with PyTorch Lightning')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training and validation')
    # 将默认学习率调高，例如 0.1（SGD 通常使用更高的学习率）
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate for SGD')
    # 恢复 momentum 参数，以适配 SGD
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--epochs_after_deficit', type=int, default=160, help='Number of epochs to normally train')
    parser.add_argument('--gamma', type=float, default=0.97, help='LR scheduler gamma')
    parser.add_argument('--project', type=str, default='CLP_SGD', help='Wandb project name')
    parser.add_argument('--run_name', type=str, default='baseline_sgd', help='Wandb run name')
    parser.add_argument('--log_dir', type=str, default='logs/', help='Directory for CSV logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/', help='Directory for model checkpoints')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loaders')
    parser.add_argument('--precision', type=str, default="16", help='Precision for mixed precision training')
    parser.add_argument('--deficit_epoch', type=int, default=0, help='Epoch to remove transform or restore labels')
    parser.add_argument('--deficit_type', type=str, default='blur', 
                        choices=['blur', 'vertical_flip', 'label_permutation', 'noise', 'none'], 
                        help='Type of deficit to apply')
    return parser.parse_args()

args = parse_args()

class DownUpSampleTransform:
    # follow paragraph 2 of section 2 in the paper
    def __call__(self, img):
        # Downsample to 8x8
        img = F.interpolate(img.unsqueeze(0), size=(8, 8), mode='bilinear', align_corners=False)
        # Upsample back to 32x32
        img = F.interpolate(img, size=(32, 32), mode='bilinear', align_corners=False)
        return img.squeeze(0)

class VerticalFlipTransform:
    def __call__(self, img):
        if torch.rand(1).item() > 0.5:
            return transforms.functional.vflip(img)
        return img

class GaussianNoiseTransform:
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        noise = torch.randn(img.size()) * self.std + self.mean
        noisy_img = img + noise
        return torch.clamp(noisy_img, 0.0, 1.0)

class PermutedLabelsDataset(datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super(PermutedLabelsDataset, self).__init__(*args, **kwargs)
        self.permutation = None

    def set_permutation(self, permutation):
        self.permutation = permutation

    def __getitem__(self, index):
        img, target = super(PermutedLabelsDataset, self).__getitem__(index)
        if self.permutation is not None:
            target = self.permutation[target]
        return img, target

# 定义数据增强流程
train_transforms_list = [
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(4/32, 4/32)),  # 最多平移4个像素
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # CIFAR-10 均值和标准差
]

# 根据 deficit_type 添加对应的变换
if args.deficit_type == 'blur':
    train_transforms_list.insert(-1, DownUpSampleTransform())  # 在归一化之前添加模糊变换
elif args.deficit_type == 'vertical_flip':
    train_transforms_list.insert(-1, VerticalFlipTransform())
elif args.deficit_type == 'noise':
    train_transforms_list.insert(-1, GaussianNoiseTransform())
# 对于 label_permutation 和 none，不需要在 transforms 中添加任何东西

train_transforms = transforms.Compose(train_transforms_list)

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

# 加载 CIFAR-10 数据集
if args.deficit_type == 'label_permutation':
    train_dataset = PermutedLabelsDataset(root='./data', train=True, download=True, transform=train_transforms)
else:
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)

val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)

# 实现标签置换逻辑（仅当 deficit_type 为 label_permutation 时）
if args.deficit_type == 'label_permutation' and isinstance(train_dataset, PermutedLabelsDataset):
    # 创建标签置换映射
    num_classes = 10
    permutation_mapping = {}
    for cls in range(num_classes):
        permuted_cls = random.randint(0, num_classes - 1)
        permutation_mapping[cls] = permuted_cls
    # 应用置换
    permuted_labels = [permutation_mapping[target] for target in train_dataset.targets]
    train_dataset.targets = permuted_labels
    print("Applied label permutation.")

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                          num_workers=args.num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                        num_workers=args.num_workers, pin_memory=True)

# 函数：修改 ResNet18
def create_modified_resnet18():
    # 加载未预训练的 ResNet18
    model = models.resnet18()
    # 修改第一个卷积层以适应 32x32 图像
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # 移除最大池化层以保留更多空间信息
    model.maxpool = nn.Identity()
    # 调整最后的全连接层以输出 CIFAR-10 的 10 个类别
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    return model

# 定义 LightningModule
class CIFAR10Classifier(LightningModule):
    def __init__(self, args):
        super().__init__()
        # 保存所有超参数
        self.save_hyperparameters(args)
        self.model = create_modified_resnet18()
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        acc = (outputs.argmax(dim=1) == targets).float().mean()
        # 记录训练损失和准确率
        self.log('train_loss', loss)
        self.log('train_acc', acc, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        acc = (outputs.argmax(dim=1) == targets).float().mean()
        # 记录验证损失和准确率
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
    def configure_optimizers(self):
        # 使用 SGD 优化器
        # 恢复 momentum、weight_decay 等参数
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay
        )
        # 保持 StepLR 调度器，或者根据需要更改
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]

# 定义回调函数以移除缺陷
class RemoveDeficitCallback(pl.Callback):
    def __init__(self, t0, deficit_type, train_dataset=None):
        super().__init__()
        self.t0 = t0
        self.deficit_type = deficit_type
        self.train_dataset = train_dataset
        self.original_targets = None

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.t0:
            if self.deficit_type in ['blur', 'vertical_flip', 'noise']:
                # 移除对应的 transform
                new_transforms = []
                for t in self.train_dataset.transform.transforms:
                    if self.deficit_type == 'blur' and isinstance(t, DownUpSampleTransform):
                        continue
                    elif self.deficit_type == 'vertical_flip' and isinstance(t, VerticalFlipTransform):
                        continue
                    elif self.deficit_type == 'noise' and isinstance(t, GaussianNoiseTransform):
                        continue
                    new_transforms.append(t)
                self.train_dataset.transform = transforms.Compose(new_transforms)
                print(f"Removed {self.deficit_type} transform at epoch {self.t0}.")
            elif self.deficit_type == 'label_permutation' and isinstance(self.train_dataset, PermutedLabelsDataset):
                # 恢复原始标签
                self.train_dataset.set_permutation(None)
                print(f"Restored original labels at epoch {self.t0}.")

# 初始化日志记录器
wandb_logger = WandbLogger(name=args.run_name, project=args.project)
csv_logger = CSVLogger(name=args.run_name, save_dir=args.log_dir)

# 初始化回调
remove_deficit_callback = RemoveDeficitCallback(args.deficit_epoch, args.deficit_type, 
                                                train_dataset if args.deficit_type != 'none' else None)

# 初始化 Trainer
trainer = pl.Trainer(
    max_epochs=args.deficit_epoch + args.epochs_after_deficit,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1 if torch.cuda.is_available() else None,
    logger=[wandb_logger, csv_logger],
    enable_checkpointing=False,  # 不保存检查点
    precision=args.precision,
    callbacks=[
        remove_deficit_callback,
    ]
)

# 初始化模型
model = CIFAR10Classifier(args)

# 开始训练
trainer.fit(model, train_loader, val_loader)
