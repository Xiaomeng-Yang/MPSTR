import torch
import argparse
from PIL import Image
from strhub.models.utils import load_from_checkpoint, parse_model_args
from strhub.data.module import SceneTextDataModule

import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Load model and image transforms
# parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
parser.add_argument('image_path', help="Path of the Image")
# parser.add_argument('save_path', help="Path to save the images")
args, unknown = parser.parse_known_args()
kwargs = parse_model_args(unknown)

parseq = load_from_checkpoint(args.checkpoint, **kwargs).eval()

img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)

ori_img = Image.open(args.image_path).convert('RGB')
# Preprocess. Model expects a batch of images with shape: (B, C, H, W)
img = img_transform(ori_img).unsqueeze(0)

length_pred, logits = parseq(img)

'''
T = len(sa_weights_list) - 1
print(T)
print(sa_weights_list)
for i in range(len(sa_weights_list)):
    print(sa_weights_list[i].shape)
print(sa_weights_list[-1])
'''
'''
attention_mask = np.zeros((T+1, 2*(T+2)))
for i in range(T+1):
    attention_mask[i][:i+1] = sa_weights_list[i][:i+1]
    if 2*(T+2) <= len(sa_weights_list[i]):
        attention_mask[i][-(T+2):] = sa_weights_list[i][(T+2):2*(T+2)]
    else:
        attention_mask[i][-(T+2):len(sa_weights_list[i])] = sa_weights_list[i][(T+2):]
'''

# logits.shape  # torch.Size([1, 26, 95]), 94 characters + [EOS] symbol
# Greedy decoding
pred = logits.softmax(-1)
label, confidence = parseq.tokenizer.decode(pred)
print('Decoded label = {}'.format(label[0]))

# print(attention_mask)
T = len(label[0])

attention_mask = sa_weights[:T+1, :T+3]


'''
for i in range(len(ca_weights_list)):
    attention_mask = np.repeat(np.repeat(ca_weights_list[i], 4, axis=0), 8, axis=1)
    # load the image
    img_h, img_w = ori_img.size[0], ori_img.size[1]
    plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))

    # scale the image
    ori_img = ori_img.resize((128, 32))
    plt.imshow(ori_img, alpha=1)
    plt.axis('off')
    
    # normalize the attention mask
    mask = cv2.resize(attention_mask, (128, 32))
    normed_mask = mask / mask.max()
    normed_mask = (normed_mask * 255).astype('uint8')
    plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap="jet")
    plt.savefig(args.save_path + str(i+1) + '.png')
'''
fig, ax = plt.subplots(figsize=(25, 18)) # set figure size
heatmap = ax.pcolor(attention_mask, cmap=plt.cm.Blues, alpha=0.9)
# print(list(label[0]))

Y_label = list(label[0]) + ['[E]']
X_label = ['[B]']*2 + list(label[0]) + ['[E]']

xticks = range(0,len(X_label))
ax.set_xticks(xticks, minor=False) # major ticks

ax.set_xticklabels(X_label, minor = False, fontsize=18)   # labels should be 'unicode'

# print(Y_label)
yticks = range(0, len(Y_label))
ax.set_yticks(yticks, minor=False)
ax.set_yticklabels(Y_label, minor = False, fontsize=18)   # labels should be 'unicode'
fig.colorbar(heatmap)

ax.grid(True)

'''
attention_mask = np.repeat(np.repeat(attention_mask, 8, axis=0), 8, axis=1)
# load the image
img_h, img_w = (T+1) * 8, 2*(T+2)*8
plt.subplots(nrows=1, ncols=1, figsize=(0.1 * img_h, 0.1 * img_w))
plt.axis('off')

# normalize the attention mask
mask = cv2.resize(attention_mask, (img_w, img_h))
# normed_mask = mask / mask.max()
# normed_mask = (normed_mask * 255).astype('uint8')
plt.imshow(attention_mask, alpha=0.5, interpolation='nearest', cmap="jet")
'''
plt.savefig(args.save_path + 'new' + '.png')
