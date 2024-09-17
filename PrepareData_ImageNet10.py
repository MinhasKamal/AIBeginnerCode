import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm


DATA_ROOT_PATH = 'D:\\Projects\\Datasets\\imagenet-10'
DATA_CHANNEL = 3 # RGB
DATA_DIM = 16 # height and width
TRAIN_TEST_SPLIT = [0.8, 0.2]
BATCH_SIZE = 64
EPOCHS = 5

torch.manual_seed(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'operation running on {device}')


##** Data Preparation **##

transform = transforms.Compose([
    transforms.Resize((DATA_DIM, DATA_DIM)),
    transforms.ToTensor()
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


dataset = datasets.ImageFolder(root=DATA_ROOT_PATH, transform=transform)
# print(len(dataset))
# print(dataset.imgs[0])

train_set, test_set = torch.utils.data.random_split(
    dataset,
    TRAIN_TEST_SPLIT) 
# print(len(train_set))
# print(len(test_set))

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last = True)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False, # no need to shuffle test data
    drop_last = True)


##** Model **##

class Model(nn.Module):
    def __init__(
            self,
            img_channel=DATA_CHANNEL,
            img_height=DATA_DIM,
            img_width=DATA_DIM,
            layer1_kernels=8,
            layer2_kernels=16,
            hidden1=120,
            hidden2=84,
            output=10):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
            in_channels=img_channel,
            out_channels=layer1_kernels,
            kernel_size=3,
            stride=1),
            nn.ReLU(),
            nn.MaxPool2d(
            kernel_size=2,
            stride=2),

            nn.Conv2d(
            in_channels=layer1_kernels,
            out_channels=layer2_kernels,
            kernel_size=3,
            stride=1),
            nn.ReLU(),
            nn.MaxPool2d(
            kernel_size=2,
            stride=2),

            nn.Flatten()
        )

        in_features = self.calc_flattened_input_dim(img_channel, img_height, img_width)

        self.fc = nn.Sequential(
            nn.Linear(
            in_features=in_features,
            out_features=hidden1,
            bias=True),
            nn.ReLU(),

            nn.Linear(
            in_features=hidden1,
            out_features=hidden2,
            bias=True),
            nn.ReLU(),

            nn.Linear(
            in_features=hidden2,
            out_features=output,
            bias=True)
            # we dont need a softmax layer as we are using cross entropy which
            # does softmax internally.
            # nn.Softmax()
        )

    def calc_flattened_input_dim(self, img_channel, img_height, img_width):
        x = torch.rand(1, img_channel, img_height, img_width)
        x = self.conv(x)
        return x.size(1)

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)


##** Train **##
model = Model().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

training_loss_list = []
testing_loss_list = []
for i in tqdm(range(EPOCHS)):
    # set model to training mode. this is ensures that batch-norm and
    # drop-out behave accordingly during training
    model.train()
    # batch training
    for j, (X_train, y_train) in enumerate(train_loader):
        # plt.imshow(X_train[0].permute(1, 2, 0))
        # plt.show()

        y_pred = model(X_train.to(device))
        batch_training_loss = criterion(y_pred, y_train.to(device))

        optimizer.zero_grad()
        batch_training_loss.backward()
        optimizer.step()

    # append loss from the last train batch for showing in plot
    training_loss_list.append(batch_training_loss.item())

    # swith to evaluation mode
    model.eval()
    number_of_correct_predictions = 0
    with torch.no_grad():
        for k, (X_test, y_test) in enumerate(test_loader):
            y_eval = model(X_test.to(device))
            _, eval = torch.max(y_eval, dim=1)
            number_of_correct_predictions += (eval == y_test.to(device)).sum()

    print(f'epoch: {i}: {number_of_correct_predictions} correct predictions in {(k+1)*BATCH_SIZE} samples')

    # append loss from the last test batch for showing in plot
    batch_testing_loss = criterion(y_eval, y_test.to(device))
    testing_loss_list.append(batch_testing_loss.item())

##** Evaluate Training **##

plt.plot(training_loss_list, label = 'training loss')
plt.plot(testing_loss_list, label = 'testing loss')
plt.legend()
plt.show()
