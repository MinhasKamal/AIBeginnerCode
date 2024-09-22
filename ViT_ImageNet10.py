import os
import torch
from torch import nn, Tensor
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import einops
from einops.layers.torch import Rearrange


##** Parameters and Settings **##

DATA_ROOT_PATH = 'D:\\Projects\\Datasets\\imagenet-10'
DATA_CHANNEL = 3 # RGB
DATA_DIM = 256 # height and width of image
CLASS_COUNT = 10
TRAIN_TEST_SPLIT = [0.8, 0.2]
BATCH_SIZE = 6
SEED = 1

PATCH_SIZE = 8
PATCH_EMBEDDING_LEN = 128 # we could take any value, but we selected
                          # value = PATCH_SIZE*PATCH_SIZE*DATA_CHANNEL
NUM_HEADS = 4
NUM_LAYERS = 4
DROPOUT = 0.1

torch.manual_seed(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'operation running on {device}')



##** Data Preparation **##

transform = transforms.Compose([
    transforms.Resize((DATA_DIM, DATA_DIM)),
    transforms.ToTensor()
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
    drop_last=True)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False, # no need to shuffle test data
    drop_last=True)
# plt.imshow(next(iter(test_loader))[0][0].permute(1, 2, 0))
# plt.show()



##** Model **##
class PatchEmbedding(nn.Sequential):
    def __init__(
            self,
            in_channel=DATA_CHANNEL, # Color channels
            patch_size=PATCH_SIZE,
            patch_embedding_len=PATCH_EMBEDDING_LEN):
        super().__init__(
            Rearrange(
                'batch color (count1 patch_height) (count2 patch_width) ->\
                batch (count1 count2) (patch_height patch_width color)',
                patch_height = patch_size,
                patch_width = patch_size
            ),
            nn.Linear(
                in_features=patch_size*patch_size*in_channel,
                out_features=patch_embedding_len
            )
        )

# patch_embedding_tmp = PatchEmbedding().forward(next(iter(test_loader))[0])
# print(patch_embedding_tmp.size()) # batch size, patch count, embed length
# disable the linear layer inside PatchEmbedding to look into the patches
# plt.imshow(patch_embedding_tmp[0][0].view(PATCH_SIZE, PATCH_SIZE, -1))
# plt.show()


class Attention(nn.Module):
    def __init__(
            self,
            embed_dim=PATCH_EMBEDDING_LEN,
            num_heads=NUM_HEADS,
            dropout=DROPOUT):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embed_dim) # TODO
        self.query = nn.Linear(in_features = embed_dim, out_features = embed_dim)
        self.key =  nn.Linear(in_features = embed_dim, out_features = embed_dim)
        self.value =  nn.Linear(in_features = embed_dim, out_features = embed_dim)
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer_norm(x) # TODO
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        attn_output, _ = self.multi_head_attention(query, key, value)
        return attn_output

# print(Attention().forward(torch.ones((1, 7, 128))).shape)


class FeedForward(nn.Sequential):
    def __init__(
            self,
            input_output_dim=PATCH_EMBEDDING_LEN,
            hidden_dim=PATCH_EMBEDDING_LEN*2,
            dropout=DROPOUT):
        super().__init__(
            nn.LayerNorm(normalized_shape=input_output_dim), # TODO
            nn.Linear(
                in_features=input_output_dim,
                out_features=hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(
                in_features=hidden_dim,
                out_features=input_output_dim),
            nn.Dropout(p=dropout)
        )

class ResedualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


# class TransformerEncoderBlock(nn.Module):
#     def __init__(
#             self,
#             in_img_channels=DATA_CHANNEL,
#             in_img_height=DATA_DIM,
#             in_img_width=DATA_DIM,
#             patch_size=PATCH_SIZE,
#             embedding_len=PATCH_EMBEDDING_LEN,
#             num_heads=NUM_HEADS,
#             num_layer=NUM_LAYERS,
#             dropout=DROPOUT,
#             out_channels=CLASS_COUNT
#             ):
#         super(self).__init__()


class ViT(nn.Module):
    def __init__(
            self,
            in_img_size=DATA_DIM,
            patch_size=PATCH_SIZE,
            embedding_len=PATCH_EMBEDDING_LEN,
            num_layers=NUM_LAYERS,
            out_channels=CLASS_COUNT
            ):
        super().__init__()

        self.patch_embedding = PatchEmbedding()
        self.class_token = nn.Parameter( # this vector will contain the extracrted feature
                # by the transformer. We could put 0 in it as value, or any random number.
                # Instead, we are making it a learnable parameter which results in better
                # performance.
                # After training, this array might represent a middle ground among the
                # classes, from where getting transformed to a specific class is easier.
            torch.rand(1, 1, embedding_len)
        )

        patch_count = (in_img_size // patch_size) ** 2 + 1 # image patches + class token
        self.positional_embedding = nn.Parameter( # we could use sine/cosine function for
                # positional embedding but we dont need that. As, the number of image
                # patches is fixed. Also, we are making them learnable parameters, which
                # reduce biases and improves performance
            torch.randn(1, patch_count, embedding_len)
        )

        self.transformer_encoder_blocks = nn.ModuleList([])
        for _ in range(num_layers):
            transformer_block = nn.Sequential(
                ResedualAdd(Attention()),
                ResedualAdd(FeedForward())
            )
            self.transformer_encoder_blocks.append(transformer_block)

        self.classification_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_len),
            nn.Linear(
                in_features=embedding_len,
                out_features=out_channels
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embedding(x)

        class_tokens = einops.repeat( # copying the current class token value
                # accroding to batch size (for each image in the batch)
            self.class_token,
            '1 1 embedding_len -> batch_size 1 embedding_len',
            batch_size = BATCH_SIZE # = len(x)
        )
        x = torch.cat( # adding the same class token (changes after each iteration)
                # infront of each image's patch-embedding-list
            tensors=(class_tokens, x),
            dim=1
        )

        x += self.positional_embedding ### DID not follow given code
                # adding 2+3=5, not concatenating or stacking 

        for transformer_encoder_block in self.transformer_encoder_blocks:
            x = transformer_encoder_block(x)

        transformed_class_tokens = x[:, 0, :]
        return self.classification_head(transformed_class_tokens) 

print(ViT().forward(torch.ones(BATCH_SIZE, DATA_CHANNEL, DATA_DIM, DATA_DIM)))


