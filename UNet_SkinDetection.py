import os
from PIL import Image
import torch
from torch import nn, Tensor
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt



##** Parameters and Settings **##

IMG_PATH = 'D:\\Projects\\Datasets\\HumanSkin\\image'
MASK_PATH = 'D:\\Projects\\Datasets\\HumanSkin\\mask'
MODEL_PATH = 'D:\\Projects\\Datasets\\UNet_SkinDetection.pth'
DATA_CHANNEL = 3 # Color image- RGB
DATA_DIM = 128 # height and width; theoritically, unet should be able
    # to work on any image-size; but getting this error: stack expects
    # each tensor to be equal size.
CLASS_COUNT = 1 # Skin
TRAIN_TEST_SPLIT = [0.9, 0.1] # 90% training data and 10% testing data
BATCH_SIZE = 6
EPOCHS = 10
LEARNING_RATE = 0.001
SEED = 1

torch.manual_seed(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'operation running on {device}')



##** DataSet **##

class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.imgs = os.listdir(img_dir)
        self.masks = os.listdir(mask_dir)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_name = self.masks[idx]
        mask_path = os.path.join(self.mask_dir, mask_name)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") # Use "L" for grayscale masks

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask



##** Data Preparation **##

transform = transforms.Compose([
    transforms.Resize((DATA_DIM, DATA_DIM)),
    transforms.ToTensor()
])

dataset = SegmentationDataset(
    img_dir=IMG_PATH,
    mask_dir=MASK_PATH,
    transform=transform)
# print(len(dataset))
# print(dataset.imgs[0])

train_set, test_set = random_split(
    dataset,
    TRAIN_TEST_SPLIT)
# print(len(train_set))
# print(len(test_set))

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last = True)
test_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False, # no need to shuffle test data
    drop_last = True)



##** Model **##

class DoubleConv(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                # the original UNet does not have padding.
                # but padding helps to simplify and ease the
                # dimention calculation for skip connections.
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

# print(DoubleConv(DATA_CHANNEL, 64).forward(
#     torch.rand((1, DATA_CHANNEL, 256, 256))).size())


class DownConv(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels):
        super().__init__()
        self.conv = DoubleConv(
            in_channels=in_channels,
            out_channels=out_channels
        )
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

    def forward(self, x: Tensor):
        conv = self.conv(x)
        pool = self.pool(conv)
        return conv, pool

# print([i.size() for i in DownConv(DATA_CHANNEL, 64).forward(
#     torch.rand((1, DATA_CHANNEL, 256, 256)))])


class UpConv(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels//2,
            kernel_size=2,
            stride=2
        )
        self.conv = DoubleConv(
            in_channels=in_channels,
            out_channels=out_channels
        )

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

# print(UpConv(128, DATA_CHANNEL).forward(
#     torch.rand((1, 128, 256//2, 256//2)),
#     torch.rand((1, 128//2, 256, 256))).size())


class UNet(nn.Module):
    def __init__(
            self,
            in_img_channels=DATA_CHANNEL,
            layer1_kernels=64,
            layer2_kernels=128,
            layer3_kernels=256,
            layer4_kernels=512,
            bottle_neck_kernels=1024,
            out_channels=CLASS_COUNT):
        super().__init__()

        self.down_conv1 = DownConv(
            in_channels=in_img_channels,
            out_channels=layer1_kernels
        )
        self.down_conv2 = DownConv(
            in_channels=layer1_kernels,
            out_channels=layer2_kernels
        )
        self.down_conv3 = DownConv(
            in_channels=layer2_kernels,
            out_channels=layer3_kernels
        )
        self.down_conv4 = DownConv(
            in_channels=layer3_kernels,
            out_channels=layer4_kernels
        )

        self.bottle_neck = DoubleConv(
            in_channels=layer4_kernels,
            out_channels=bottle_neck_kernels
        )

        self.up_conv4 = UpConv(
            in_channels=bottle_neck_kernels,
            out_channels=layer4_kernels
        )
        self.up_conv3 = UpConv(
            in_channels=layer4_kernels,
            out_channels=layer3_kernels
        )
        self.up_conv2 = UpConv(
            in_channels=layer3_kernels,
            out_channels=layer2_kernels
        )
        self.up_conv1 = UpConv(
            in_channels=layer2_kernels,
            out_channels=layer1_kernels
        )

        self.out = nn.Conv2d(
            in_channels=layer1_kernels,
            out_channels=out_channels,
            kernel_size=1
        )

    def forward(self, x: Tensor) -> Tensor:
        down1, pool1 = self.down_conv1(x)
        down2, pool2 = self.down_conv2(pool1)
        down3, pool3 = self.down_conv3(pool2)
        down4, pool4 = self.down_conv4(pool3)

        bottle_neck = self.bottle_neck(pool4)

        up4 = self.up_conv4(bottle_neck, down4)
        up3 = self.up_conv3(up4, down3)
        up2 = self.up_conv2(up3, down2)
        up1 = self.up_conv1(up2, down1)

        out = self.out(up1)
        return out

# print(UNet().forward(
#     torch.rand((1, DATA_CHANNEL, 256, 256))).size())



##** Training **##
model = UNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
print(f'loaded pretrained model from {MODEL_PATH}')

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

training_loss_list = []
testing_loss_list = []
for i in tqdm(range(EPOCHS)):
    # set model to training mode. this is ensures that batch-norm and
    # drop-out behave accordingly during training
    model.train()
    # batch training
    batch_training_loss_sum = 0
    for j, (X_train, y_train) in enumerate(train_loader):
        # plt.imshow(X_train[0].permute(1, 2, 0))
        # plt.show()
        # plt.imshow(y_train[0].permute(1, 2, 0))
        # plt.show()

        y_pred = model(X_train.to(device))
        # plt.imshow(y_pred[0].detach().permute(1, 2, 0))
        # plt.show()
        optimizer.zero_grad()
        batch_training_loss = criterion(y_pred, y_train.to(device))
        batch_training_loss_sum += batch_training_loss.item()

        batch_training_loss.backward()
        optimizer.step()

    # append average loss
    training_loss_list.append(batch_training_loss_sum / (j+1))

    # swith to evaluation mode
    model.eval()
    batch_testing_loss_sum = 0
    # number_of_correct_predictions = 0
    with torch.no_grad():
        for k, (X_test, y_test) in enumerate(test_loader):
            y_eval = model(X_test.to(device))
            batch_testing_loss = criterion(y_eval, y_test.to(device))
            batch_testing_loss_sum += batch_testing_loss.item()

            # plt.imshow(X_test[0].permute(1, 2, 0))
            # plt.show()
            # plt.imshow(y_eval[0].permute(1, 2, 0), cmap='grey')
            # plt.show()

            # _, eval = torch.max(y_eval, dim=1)
            # number_of_correct_predictions += (eval == y_test.to(device)).sum()

    # print(f'epoch: {i}: {number_of_correct_predictions} correct predictions in {(k+1)*BATCH_SIZE} samples')

    # append average test-loss
    testing_loss_list.append(batch_testing_loss_sum / (k+1))

print(training_loss_list)
print(testing_loss_list)

torch.save(model.state_dict(), MODEL_PATH)



##** Evaluation **##

plt.plot(training_loss_list, label='training loss')
plt.plot(testing_loss_list, label='testing loss')
plt.legend()
plt.show()


plt.imshow(X_test[0].permute(1, 2, 0))
plt.show()
plt.imshow(y_eval[0].permute(1, 2, 0), cmap='grey')
plt.show()

