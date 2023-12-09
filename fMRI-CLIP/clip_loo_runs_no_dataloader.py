#This script is designed to run on the Compute Canada remote computers using Python version 3.7.9

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

# device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

NUM_WORDS = 20
OVERLAPPING = True
NUM_SUBJS = 8
subjects_fmri = [] #stores all 8 subject fmri np arrays
fMRI_folder = Path('./doi_10.5061_dryad.gt413__v1')

with open(fMRI_folder / 'fmri_indices', 'rb') as f:
    fmri_indices = pickle.load(f)

assert fMRI_folder.exists(), f"Foldder: {fMRI_folder} does not exist."
for subj_id in range(8): 
    fmri_file_name = str(subj_id) + '_smooth_detrend_nifti_4d.nii'
    fmri = nib.load(fMRI_folder / fmri_file_name)
    fmri = np.array(fmri.dataobj)
    assert isinstance(fmri, np.ndarray), f"Imported fmri_scan for subject {subj_id} is not of type numpy.ndarray"
    assert(fmri.ndim) == 4, f"Imported fmri_scan for subject {subj_id} is not 4 dimensional"
    subjects_fmri.append(fmri)

words_info = [] #stores tuples of (word, time, features) sorted by time appeared
mat_file = fMRI_folder / 'subject_1.mat' #only looks at the first subject file, somewhere it said all the timings were the same so this should be safe
mat_contents = sio.loadmat(mat_file)
for count, row in enumerate(mat_contents['words'][0]):
    word_value = row[0][0][0][0]
    time = row[1][0][0]
    word_tuple = (word_value, time)
    words_info.append(word_tuple)

subjects_samples = [[[] for j in range(4)] for i in range(NUM_SUBJS)] #stores lists of all the samples for each subject and for each of the 4 runs
window = signal.windows.gaussian(16, std=1) #gaussian window for the 4 fMRI scans
print("NUM_WORDS:", NUM_WORDS)
word_count = 0
new_run = True
run_count = 0
while word_count < len(words_info) - NUM_WORDS:
    #gets the 4 input words, and the 4 consecutive words while verifying they were read in sequence
    scan_words = []
    start_time = words_info[word_count][1]
    valid_word = True #tracks if the words are in sequence or not
    all_weighted_word_scans = []
    for i in range(NUM_WORDS):
        word_info = words_info[word_count + i]
        if word_info[1] != start_time + 0.5*i:
            #if some of the words are not in sequence, skip forward 1 word after innter loop
            valid_word = False
            break
        scan_words.append(word_info[0])
        fmri_count = 0
        weighted_word_scans = [] #contains weighted word scans for each subject
        for j in range(1,17):
            delay = 0.5*j #time after word was read
            try:
                curr_fmri_idx = fmri_indices.index((start_time + delay)/2) #checks if an fMRI scan happens at this time point
                weight = window[int(2*delay)-1]
                for count, subject in enumerate(subjects_fmri):
                    if fmri_count == 0:
                        weighted_word_scans.append(weight*subject[:,:,:,curr_fmri_idx])
                    else:
                        weighted_word_scans[count] += weight*subject[:,:,:,curr_fmri_idx]
                fmri_count += 1
            except Exception as e:
                pass
        if fmri_count != 4:
            if new_run == True:
                print("Run ended:", start_time)
                new_run = False
                run_count += 1
            valid_word = False
            break
        all_weighted_word_scans.append(weighted_word_scans)
    if not valid_word:
        word_count +=1
        continue
    for subject_count in range(NUM_SUBJS):
        for word_scan_count, weighted_scans in enumerate(all_weighted_word_scans):
            if word_scan_count == 0:
                summed_weighted_scan = weighted_scans[subject_count]
            else:
                summed_weighted_scan += weighted_scans[subject_count]
        subjects_samples[subject_count][run_count].append((summed_weighted_scan, scan_words))
    new_run = True
    # print("Created sample:")
    # print("\tScan time:", str(start_time))
    # print("\tInput words:", str(scan_words))
    #if successful, skip forward to the next set of 4 words
    if OVERLAPPING:
        word_count += 4
    else:
        word_count += NUM_WORDS

print("Run_count:", run_count)

print("Total number of samples:", str(len(subjects_samples[0][0]) + len(subjects_samples[0][1]) + len(subjects_samples[0][2]) + len(subjects_samples[0][3])))

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

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv3d(1, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
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
            width=vision_width
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
        self.logit_scale = torch.ones([]) * np.log(1 / 0.07)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):

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

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

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

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

#trains the clip model from scratch
def train_clip(model, images, text, batch_size=10, num_epochs=100, lr=1e-3, debug=False):
    print("Training...")
    for epoch in range(num_epochs):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.2)
        loss_fn = nn.CrossEntropyLoss()
        model.zero_grad(set_to_none=True)
        epoch_loss = 0
        image_epoch_correct = 0
        text_epoch_correct = 0
        epoch_total = 0
        p = torch.randperm(len(images))
        epoch_text_samples, epoch_image_samples = (text.cpu())[p].to(device), (images.cpu())[p].to(device)    
        for batch in range(math.floor(epoch_image_samples.shape[0]/batch_size)):
            optimizer.zero_grad()
            
            #gets embeddings for text and image batches
            start_idx = batch*batch_size
            end_idx = (batch+1)*batch_size
            text_batch, image_batch = epoch_text_samples[start_idx:end_idx], epoch_image_samples[start_idx:end_idx]
            
            logits_per_image, logits_per_text = model(torch.unsqueeze(image_batch, dim=1), text_batch)
            
            #symmetric loss function
            labels = torch.arange(batch_size).to(device)
            loss_text = loss_fn(logits_per_text, labels)
            loss_image = loss_fn(logits_per_image, labels)
            loss = (loss_text + loss_image)/2
            loss.backward()
            epoch_loss += loss.detach().item()
            optimizer.step()
            
            #compute accuracy
            image_winners = torch.argmax(logits_per_image, dim=0)
            text_winners = torch.argmax(logits_per_text, dim=0)
            image_corrects = (image_winners == labels)
            text_corrects = (text_winners == labels)
            total_image_correct = image_corrects.sum().float().item()
            total_text_correct = text_corrects.sum().float().item()
            image_epoch_correct += total_image_correct
            text_epoch_correct += total_text_correct
            epoch_total += batch_size

            del logits_per_image
            del logits_per_text
            del text_batch
            del image_batch
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
        if debug:
            print("\tEpoch:", epoch, "Training Loss:", epoch_loss, "Training Image Accuracy:", image_epoch_correct/epoch_total, "Training Text Accuracy:", text_epoch_correct/epoch_total)
        del epoch_text_samples
        del epoch_image_samples
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

def split_subject_samples(subjects_samples, min_val=None, max_val=None, num_tokens=4*NUM_WORDS):
    images = torch.zeros([len(subjects_samples[0])] + list(subjects_samples[0][0][0].shape))
    text = torch.zeros((len(subjects_samples[0]), num_tokens), dtype=int)
    for idx in range(len(subjects_samples[0])):
        for subj_id, samples in enumerate(subjects_samples):
            if subj_id == 0:
                avg_fmri = samples[idx][0]
            else:
                avg_fmri += samples[idx][0]
        images[idx] = torch.tensor(avg_fmri/len(subjects_samples[0]))
        text[idx] = tokenize([" ".join(subjects_samples[0][idx][1])], context_length=num_tokens)
    if min_val is None:
        min_val = images.min()
        max_val = images.max()
    images = (images - min_val)/(max_val - min_val)
    return images.to(device), text.to(device), min_val, max_val

def split_samples(samples, min_val=None, max_val=None, num_tokens=4*NUM_WORDS):
    images = torch.zeros([len(samples)] + list(samples[0][0].shape))
    text = torch.zeros((len(samples), num_tokens), dtype=int)
    for idx, sample in enumerate(samples):
        images[idx] = torch.tensor(sample[0])
        text[idx] = tokenize([" ".join(sample[1])], context_length=num_tokens)
    if min_val is None:
        min_val = images.min()
        max_val = images.max()
    images = (images - min_val)/(max_val - min_val)
    return images.to(device), text.to(device), min_val, max_val

def assess_accuracy(model, images, text, batch_size=32):
    total_samples = 0
    total_image_correct = 0
    total_text_correct = 0
    total_loss = 0

    loss_fn = nn.CrossEntropyLoss()

    p = torch.randperm(len(images))
    temp_text_samples, temp_image_samples = (text.cpu())[p].to(device), (images.cpu())[p].to(device)   

    for batch in range(math.floor(temp_image_samples.shape[0]/batch_size)):
                
        #gets embeddings for text and image batches
        start_idx = batch*batch_size
        end_idx = (batch+1)*batch_size
        text_batch, image_batch = temp_text_samples[start_idx:end_idx], temp_image_samples[start_idx:end_idx]

        logits_per_image, logits_per_text = model(torch.unsqueeze(image_batch, dim=1), text_batch)
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
    del temp_text_samples
    del temp_image_samples
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return total_loss, total_image_correct/total_samples, total_text_correct/total_samples

NUM_EVAL_ITERS = 10

for subj_id in range(len(subjects_samples)):
    print("Subject:", subj_id)
    loo_full_batch_img_train_total = 0
    loo_full_batch_text_train_total = 0
    loo_full_batch_img_test_total = 0
    loo_full_batch_text_test_total = 0
    loo_two_batch_img_train_total = 0
    loo_two_batch_text_train_total = 0
    loo_two_batch_img_test_total = 0
    loo_two_batch_text_test_total = 0
    for run in range(4):
        print("Run:", run)
        train_samples = [] #gets 3 runs besides current left out one
        test_samples = None
        for i in range(len(subjects_samples[0])): #use first 3 subjects for training
            if i != run:
                train_samples += subjects_samples[subj_id][i]
            else:
                test_samples = subjects_samples[subj_id][i]
        print("Training samples length:", len(train_samples))
        print("Testing samples length:", len(test_samples))
        #use all subjects except one for training
        train_images, train_text, min_val, max_val = split_samples(train_samples)
        test_images, test_text, _, _ = split_samples(test_samples, min_val, max_val)

        clip_model = CLIP(
            1024, #embed dim
            image_resolution, 
            (2,2,2,2), #image encoder layers 
            64, #image encoder width
            4*NUM_WORDS, #token context length
            vocab_size, 
            transformer_width, 
            transformer_heads, 
            transformer_layers
        ).to(device)

        train_clip(clip_model, train_images, train_text, batch_size=32, lr=1e-5, num_epochs=10, debug=True)

        torch.save(clip_model.state_dict(), "./saved_clip_model_subj_" + str(subj_id) + "_minus_" + str(run) + "_words_" + str(NUM_WORDS) + "_no_overlap")
        print("Saved file: ./saved_clip_model_subj_" + str(subj_id) + "minus_" + str(run) + "_words_" + str(NUM_WORDS) + "_overlap")

        full_batch_img_train_total = 0
        full_batch_text_train_total = 0
        full_batch_img_test_total = 0
        full_batch_text_test_total = 0
        two_batch_img_train_total = 0
        two_batch_text_train_total = 0
        two_batch_img_test_total = 0
        two_batch_text_test_total = 0

        for i in range(NUM_EVAL_ITERS):
            print("\tIteration:", i)

            train_loss, train_image_acc, train_text_acc = assess_accuracy(clip_model, train_images, train_text, batch_size=32)
            full_batch_img_train_total += train_image_acc
            full_batch_text_train_total += train_text_acc
            print("\t", 32, "Batch Train Loss:", train_loss, ", Image Accuracy:", train_image_acc, ", Text Accuracy:", train_text_acc)

            test_loss, test_image_acc, test_text_acc = assess_accuracy(clip_model, test_images, test_text, batch_size=32)
            full_batch_img_test_total += test_image_acc
            full_batch_text_test_total += test_text_acc
            print("\t", 32, "Batch Test Loss:", test_loss, ", Image Accuracy:", test_image_acc, ", Text Accuracy:", test_text_acc)

            train_loss, train_image_acc, train_text_acc = assess_accuracy(clip_model, train_images, train_text, batch_size=2)
            two_batch_img_train_total += train_image_acc
            two_batch_text_train_total += train_text_acc
            print("\t", 2, "Batch Train Loss:", train_loss, ", Image Accuracy:", train_image_acc, ", Text Accuracy:", train_text_acc)

            test_loss, test_image_acc, test_text_acc = assess_accuracy(clip_model, test_images, test_text, batch_size=2)
            two_batch_img_test_total += test_image_acc
            two_batch_text_test_total += test_text_acc
            print("\t", 2, "Batch Test Loss:", test_loss, ", Image Accuracy:", test_image_acc, ", Text Accuracy:", test_text_acc)

        print("\tAverage full batch train image accuracy:", full_batch_img_train_total/NUM_EVAL_ITERS)
        print("\tAverage full batch train text accuracy:", full_batch_text_train_total/NUM_EVAL_ITERS)
        print("\tAverage full batch test image accuracy:", full_batch_img_test_total/NUM_EVAL_ITERS)
        print("\tAverage full batch test text accuracy:", full_batch_text_test_total/NUM_EVAL_ITERS)
        print("\tAverage two batch train image accuracy:", two_batch_img_train_total/NUM_EVAL_ITERS)
        print("\tAverage two batch train text accuracy:", two_batch_text_train_total/NUM_EVAL_ITERS)
        print("\tAverage two batch test image accuracy:", two_batch_img_test_total/NUM_EVAL_ITERS)
        print("\tAverage two batch test text accuracy:", two_batch_text_test_total/NUM_EVAL_ITERS)

        del train_images
        del train_text
        del test_images
        del test_text
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        loo_full_batch_img_train_total += full_batch_img_train_total/NUM_EVAL_ITERS
        loo_full_batch_text_train_total += full_batch_text_train_total/NUM_EVAL_ITERS
        loo_full_batch_img_test_total += full_batch_img_test_total/NUM_EVAL_ITERS
        loo_full_batch_text_test_total += full_batch_text_test_total/NUM_EVAL_ITERS
        loo_two_batch_img_train_total += two_batch_img_train_total/NUM_EVAL_ITERS
        loo_two_batch_text_train_total += two_batch_text_train_total/NUM_EVAL_ITERS
        loo_two_batch_img_test_total += two_batch_img_test_total/NUM_EVAL_ITERS
        loo_two_batch_text_test_total += two_batch_text_test_total/NUM_EVAL_ITERS

    print("Average full batch train image accuracy:", loo_full_batch_img_train_total/4)
    print("Average full batch train text accuracy:", loo_full_batch_text_train_total/4)
    print("Average full batch test image accuracy:", loo_full_batch_img_test_total/4)
    print("Average full batch test text accuracy:", loo_full_batch_text_test_total/4)
    print("Average two batch train image accuracy:", loo_two_batch_img_train_total/4)
    print("Average two batch train text accuracy:", loo_two_batch_text_train_total/4)
    print("Average two batch test image accuracy:", loo_two_batch_img_test_total/4)
    print("Average two batch test text accuracy:", loo_two_batch_text_test_total/4)
