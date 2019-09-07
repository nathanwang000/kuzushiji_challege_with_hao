# ---------------------------------------------------------------------------- #
# An implementation of https://arxiv.org/pdf/1512.03385.pdf                    #
# See section 4.2 for the model architecture on CIFAR-10                       #
# Some part of the code was referenced from below                              #
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py   #
# ---------------------------------------------------------------------------- #

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import MTL_OPT.lib.optimizer as optimizers
from MTL_OPT.lib.criterion import L2
from MTL_OPT.lib.utils import AverageMeter, OptRecorder, random_string
from MTL_OPT.lib.utils import random_split_dataset
import argparse, tqdm
from sklearn.externals import joblib
import random, string, os
parser = argparse.ArgumentParser(description="opt")
parser.add_argument('-o', type=str,
                    help='optimizer', default='optimizers.Diff')
parser.add_argument('-seed', type=int,
                    help='random seed', default=42)
parser.add_argument('-epoch', type=int,
                    help='#epoch to run', default=200)
parser.add_argument('-s', type=str,
                    help='save directory', default='train_loss')
parser.add_argument('-lr', type=float,
                    help='learning rate', default=1e-3)

args = parser.parse_args()
print(args)
train_losses = []
train_errors = []
val_errors = []
test_errors = []
torch.set_num_threads(1)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.system('mkdir -p {}'.format(args.s))
run_id = random_string()

def eval_loader(model, loader):
    model.eval()
    loss_meter = AverageMeter()
    error_meter = AverageMeter()
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm.tqdm(loader)):  
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            bs = labels.size(0)
            
            outputs = model(images)
            loss = criterion(outputs, labels) + l2loss.loss()
            loss_meter.update(loss.item(), bs)

            _, predicted = torch.max(outputs.data, 1)
            n_correct = (predicted == labels).sum().item()
            error_meter.update(0, n_correct)
            error_meter.update(1, bs-n_correct)
            
    model.train()
    return loss_meter.avg, error_meter.avg

def save(acc=-1):
    # Save the model checkpoint
    name =  "{}/{}-{}^{:.2f}^{}".format(args.s,
                                        args.o.split('.')[-1],
                                        args.lr,
                                        -1, # placeholder
                                        run_id)

    if os.path.exists('{}.train_losses'.format(name)):
        os.system('rm {}.train_losses'.format(name))
        os.system('rm {}.train_errors'.format(name))
        os.system('rm {}.val_errors'.format(name))
        os.system('rm {}.test_errors'.format(name))
        os.system('rm {}.opt_track'.format(name))
        os.system('rm {}.ckpt'.format(name))

    name =  "{}/{}-{}^{:.2f}^{}".format(args.s,
                                        args.o.split('.')[-1],
                                        args.lr,
                                        acc,
                                        run_id)

    joblib.dump(train_losses, name + ".train_losses")
    joblib.dump(train_errors, name + ".train_errors")
    joblib.dump(val_errors, name + ".val_errors")
    joblib.dump(test_errors, name + ".test_errors")
    joblib.dump(opt_recorder.tracker, name + ".opt_track")
    torch.save(model.state_dict(), name + '.ckpt')

# Hyper-parameters
num_epochs = args.epoch
batch_size = 128
learning_rate = args.lr

# Image preprocessing modules
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                             train=True, 
                                             transform=transform,
                                             download=True)
train_dataset, val_dataset = random_split_dataset(train_dataset, [0.8, 0.2],
                                                  seed=args.seed)
test_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                            train=False, 
                                            transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                         batch_size=batch_size, 
                                         shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)
        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
model = ResNet(ResidualBlock, [2, 2, 2]).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
if '(' in args.o:
    opt = args.o
    alpha_index = opt.find('(')
    alphas = eval(opt[alpha_index:])
    optimizer = eval(opt[:alpha_index])(model.parameters(), lr=learning_rate,
                                        alphas=alphas)
else:
    optimizer = eval(args.o)(model.parameters(), lr=learning_rate)
opt_recorder = OptRecorder(optimizer)
l2loss = L2(optimizer, 0.5 * 1e-4) # follow quasi-paper
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                       verbose=True)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):

    loss_meter = AverageMeter()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        def closure():
            optimizer.zero_grad()        
            outputs = model(images)
            loss = criterion(outputs, labels) + l2loss.loss()
            loss.backward()
            return loss
        
        loss = optimizer.step(closure)
        if type(loss) is list:
            for l in loss:
                loss_meter.update(l.item())
        else:
            loss_meter.update(loss.item())
        
        if (i+1) % int(total_step / 5) == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss_meter.avg))


    tr_loss, tr_error = eval_loader(model, train_loader)
    _, val_error = eval_loader(model, val_loader)
    scheduler.step(val_error)
    _, test_error = eval_loader(model, test_loader)
    opt_recorder.record()    
    print('train loss: {:.4f}, errors(tr, val, te): ({:.4f}, {:.4f}, {:.4f})'.\
          format(tr_loss, tr_error, val_error, test_error))

    train_losses.append(tr_loss)
    train_errors.append(tr_error)    
    val_errors.append(val_error)
    test_errors.append(test_error)
    save()
            
# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

acc = correct / total * 100
save(acc)
