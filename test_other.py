#!/usr/bin/env python3
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import string
import sys
import os
from dataclasses import dataclass
from nltk import edit_distance
from typing import List

import torch
from torchvision import utils as vutils

from tqdm import tqdm

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--data_root', default='/home/test13/yxm/data/chinese_scene')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cased', action='store_true', default=False, help='Cased comparison')
    parser.add_argument('--punctuation', action='store_true', default=False, help='Check punctuation')
    parser.add_argument('--rotation', type=int, default=0, help='Angle of rotation (counter clockwise) in degrees.')
    parser.add_argument('--store_result', action='store_true', default=False, help='Whether store the recognition results')
    parser.add_argument('--dir_name', type=str, default='mpstr_curve')
    parser.add_argument('--device', default='cuda')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    
    charset_test = string.digits + string.ascii_lowercase
    if args.cased:
        charset_test += string.ascii_uppercase
    if args.punctuation:
        charset_test += string.punctuation
    kwargs.update({'charset_test': charset_test})

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    hp = model.hparams
    datamodule = SceneTextDataModule(args.data_root, '_unused_', hp.img_size, hp.max_label_length, hp.charset_train,
                                     hp.charset_test, args.batch_size, args.num_workers, False, rotation=args.rotation)

    dataloader = datamodule.test_dataloader_other()
    total = 0
    correct = 0
    ned = 0
    label_length = 0
    idx = 0
        
    if args.store_result:
        output_dir = os.path.join('./visualize/union/', args.dir_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = open(os.path.join(output_dir, 'result.txt'), 'w', encoding='utf-8')
        output_file.write('GT\tPredict\tPre_Len\n')

    # for ori_imgs, ori_label, imgs, labels in tqdm(iter(dataloader)):
    for imgs, labels in tqdm(iter(dataloader)):
        res = model.test_step((imgs.to(model.device), labels), -1)['output']
        total += res.num_samples
        correct += res.correct
        ned += res.ned
        label_length += res.label_length
        
        if args.store_result:
            for i in range(len(res.pred_list)):
                # print(res.gt_list[i] + '\t' + res.pred_list[i] + '\n')
                output_file.write(res.gt_list[i] + '\t' + res.pred_list[i] + '\t' + str(res.length_list[i]) + '\n')
                if res.gt_list[i] != res.pred_list[i]:
                    # output_file.write(res.gt_list[i] + '\t' + res.pred_list[i] + '\n')
                    img_name = str(idx) + '-' + res.gt_list[i] + '-' + res.pred_list[i] + '-' + str(res.length_list[i]) + '.png'
                    img_path = os.path.join(output_dir, img_name)
                    cur_image = imgs[i].to(torch.device('cpu'))
                    vutils.save_image(cur_image, img_path)
                
                idx += 1

    accuracy = 100 * correct / total
    mean_ned = 100 * (1 - ned / total)
    mean_label_length = label_length / total

    with open(args.checkpoint + '.log.txt', 'w') as f:
        f.write('Accuracy: '+ str(accuracy))

    print('Accuracy:', accuracy)
    print('NED:', mean_ned)
    print('Label_Length:', mean_label_length)
        
if __name__ == '__main__':
    main()
