import transformers
from transformers import CLIPConfig, CLIPModel, CLIPProcessor, CLIPImageProcessor, CLIPTokenizerFast
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import random
import math
import scipy.io as sio
from scipy import signal
import nibabel as nib
from pathlib import Path
from gensim.models import Word2Vec
import re
import gc
import pickle
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

NUM_SUBJS = 8
NUM_WORDS = 4
BATCH_SIZE = 32
LEFT_OUT_SUBJ = 0
MODEL_PATH = "./saved_clip_model_minus_subj_0_words_4_concat"

class SubjectImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, num_words):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.num_words = num_words

    def __len__(self):
        return math.floor(len(self.img_labels)/4) - NUM_WORDS//4 + 1

    def __getitem__(self, idx):
        full_image = None
        full_word = None
        for word_count in range(self.num_words):
            word_idx = idx*4 + word_count
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[word_idx, 0])
            image = torch.load(img_path)
            word = self.img_labels.iloc[word_idx, 1]
            if word_count == 0:
                full_image = torch.unsqueeze(image,0)
                full_word = word
            else:
                full_image = torch.cat((full_image, torch.unsqueeze(image,0))) #adds fmri channel for each word
                #full_image += torch.unsqueeze(image,0) #sums up fmri scans from each word
                full_word += " " + word
        return full_image, full_word

import gzip
import html
import os
from functools import lru_cache
import ftfy
import regex as re

@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath("./doi_10.5061_dryad.gt413__v1")), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text
    
import hashlib
import os
import urllib
import warnings
from typing import Any, Union, List
from pkg_resources import packaging

_tokenizer = SimpleTokenizer()

def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv3d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool3d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv3d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool3d(stride)),
                ("0", nn.Conv3d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm3d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out

class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64, num_channels=1):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv3d(num_channels, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(width // 2, width // 2, kernel_size=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv3d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool3d(2)
        self.linear = nn.Linear(2048, output_dim)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
#         self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
#         x = self.avgpool(x) #changed final attentionpool to avgpool
        #x = self.attnpool(x)
        x = x.view(-1,2048) #will have to change this if different layer/kernel sizes used
        x = self.linear(x)
        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 num_channels: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        #initializes resnet (removed option for vision transformer)
        vision_heads = vision_width * 32 // 64
        self.visual = ModifiedResNet(
            layers=vision_layers,
            output_dim=embed_dim,
            heads=vision_heads,
            input_resolution=image_resolution,
            width=vision_width,
            num_channels=num_channels
        )

        #initializes text transformer
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        #self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale = torch.ones([]) * np.log(1 / 0.07)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
#             if self.visual.attnpool is not None:
#                 std = self.visual.attnpool.c_proj.in_features ** -0.5
#                 nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
#                 nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
#                 nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
#                 nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        #print("encoding images")
        image_features = self.encode_image(image)
        #print("encoding text")
        text_features = self.encode_text(text)

        # normalized features
        #print("normalizing images")
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        #print("normalizing text")
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        #print("getting logits_per_image")
        logits_per_image = logit_scale * image_features @ text_features.t()
        #print("getting logits_per_text")
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv3d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

model_path = "./RN50.pt"
jit=True
with open(model_path, 'rb') as opened_file:
    try:
        # loading JIT archive
        model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
        state_dict = model.state_dict()
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(opened_file, map_location="cpu")

#vision
counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
vision_layers = tuple(counts)
vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
image_resolution = output_width * 32

#transformer
embed_dim = state_dict["text_projection"].shape[1]
context_length = state_dict["positional_embedding"].shape[0]
vocab_size = state_dict["token_embedding.weight"].shape[0]
transformer_width = state_dict["ln_final.weight"].shape[0]
transformer_heads = transformer_width // 64
transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

# def assess_accuracy(model, text_samples, image_samples, batch_size=64):
def assess_accuracy(model, dataloaders, batch_size):
    total_samples = 0
    total_image_correct = 0
    total_text_correct = 0
    total_loss = 0

    loss_fn = nn.CrossEntropyLoss()

    for subject_data in dataloaders:
        for batch in range(math.floor(len(dataloaders[0].dataset)/batch_size)):
            images, text = next(iter(subject_data))
            image_batch = images.to(device)
            text_tokens = []
            for text_sample in text:
                text_tokens.append(tokenize(text_sample, context_length=NUM_WORDS*4))
            text_batch = torch.cat(text_tokens, dim=0).long().to(device)

            logits_per_image, logits_per_text = model(image_batch, text_batch)
            #gets random results
            # logits_per_image = torch.rand((batch_size, batch_size)).to(device)
            # logits_per_text = torch.rand((batch_size, batch_size)).to(device)

            #symmetric loss function
            labels = torch.arange(batch_size).to(device)
            loss_text = loss_fn(logits_per_text, labels).detach()
            loss_image = loss_fn(logits_per_image, labels).detach()
            loss = (loss_text + loss_image)/2
            total_loss += loss.item()
            
            #compute accuracy
            image_winners = torch.argmax(logits_per_image, dim=0)
            text_winners = torch.argmax(logits_per_text, dim=0)
            image_corrects = (image_winners == labels)
            text_corrects = (text_winners == labels)
            image_correct_num = image_corrects.sum().float().detach().item()
            text_correct_num = text_corrects.sum().float().detach().item()
            total_image_correct += image_correct_num
            total_text_correct += text_correct_num
            total_samples += batch_size

            del logits_per_image
            del logits_per_text
            del text_batch
            del image_batch
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return total_loss, total_image_correct/total_samples, total_text_correct/total_samples

def get_dataloaders(batch_size):
    train_dataloaders = []
    test_dataloader = None
    for i in range(NUM_SUBJS):
        label_filename = "./" + str(i) + "_subject_word_fmri_labels.csv"
        dataset = SubjectImageDataset(label_filename, "", NUM_WORDS)
        if i != LEFT_OUT_SUBJ:
            train_dataloaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=True))
        else:
            test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_dataloaders, test_dataloader

clip_model = CLIP(
    1024, #embed dim
    image_resolution, 
    (2,2,2,2), #image encoder layers 
    64, #image encoder width,
    NUM_WORDS, #number of fmri channels per sample
    4*NUM_WORDS, #token context length
    vocab_size, 
    transformer_width, 
    transformer_heads, 
    transformer_layers
).to(device)
clip_model.load_state_dict(torch.load(MODEL_PATH))
clip_model.eval()

full_batch_img_train_total = 0
full_batch_text_train_total = 0
full_batch_img_test_total = 0
full_batch_text_test_total = 0
two_batch_img_train_total = 0
two_batch_text_train_total = 0
two_batch_img_test_total = 0
two_batch_text_test_total = 0

full_batch_train_dataloaders, full_batch_test_dataloader = get_dataloaders(BATCH_SIZE)
print("Total number of samples:", len(full_batch_test_dataloader.dataset))

two_batch_train_dataloaders, two_batch_test_dataloader = get_dataloaders(2)
print("Total number of samples:", len(two_batch_test_dataloader.dataset))

for i in range(10):
    print("Iteration:", i)

    train_loss, train_image_acc, train_text_acc = assess_accuracy(clip_model, full_batch_train_dataloaders, BATCH_SIZE)
    full_batch_img_train_total += train_image_acc
    full_batch_text_train_total += train_text_acc
    print("\t", BATCH_SIZE, "Batch Train Loss:", train_loss, ", Image Accuracy:", train_image_acc, ", Text Accuracy:", train_text_acc)

    test_loss, test_image_acc, test_text_acc = assess_accuracy(clip_model, [full_batch_test_dataloader], BATCH_SIZE)
    full_batch_img_test_total += test_image_acc
    full_batch_text_test_total += test_text_acc
    print("\t", BATCH_SIZE, "Batch Test Loss:", test_loss, ", Image Accuracy:", test_image_acc, ", Text Accuracy:", test_text_acc)

    train_loss, train_image_acc, train_text_acc = assess_accuracy(clip_model, two_batch_train_dataloaders, 2)
    two_batch_img_train_total += train_image_acc
    two_batch_text_train_total += train_text_acc
    print("\t", 2, "Batch Train Loss:", train_loss, ", Image Accuracy:", train_image_acc, ", Text Accuracy:", train_text_acc)

    test_loss, test_image_acc, test_text_acc = assess_accuracy(clip_model, [two_batch_test_dataloader], 2)
    two_batch_img_test_total += test_image_acc
    two_batch_text_test_total += test_text_acc
    print("\t", 2, "Batch Test Loss:", test_loss, ", Image Accuracy:", test_image_acc, ", Text Accuracy:", test_text_acc)

print("Average full batch train image accuracy:", full_batch_img_train_total/10)
print("Average full batch train text accuracy:", full_batch_text_train_total/10)
print("Average full batch test image accuracy:", full_batch_img_test_total/10)
print("Average full batch test text accuracy:", full_batch_text_test_total/10)
print("Average two batch train image accuracy:", two_batch_img_train_total/10)
print("Average two batch train text accuracy:", two_batch_text_train_total/10)
print("Average two batch test image accuracy:", two_batch_img_test_total/10)
print("Average two batch test text accuracy:", two_batch_text_test_total/10)



