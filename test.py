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
from thop import profile
from thop import clever_format
from typing import List

import torch
import time

from tqdm import tqdm
from torchvision import utils as vutils

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args


@dataclass
class Result:
    dataset: str
    num_samples: int
    accuracy: float
    ned: float
    confidence: float
    label_length: float


def print_results_table(results: List[Result], file=None):
    w = max(map(len, map(getattr, results, ['dataset'] * len(results))))
    w = max(w, len('Dataset'), len('Combined'))
    print('| {:<{w}} | # samples | Accuracy | 1 - NED | Confidence | Label Length |'.format('Dataset', w=w), file=file)
    print('|:{:-<{w}}:|----------:|---------:|--------:|-----------:|-------------:|'.format('----', w=w), file=file)
    c = Result('Combined', 0, 0, 0, 0, 0)
    for res in results:
        c.num_samples += res.num_samples
        c.accuracy += res.num_samples * res.accuracy
        c.ned += res.num_samples * res.ned
        c.confidence += res.num_samples * res.confidence
        c.label_length += res.num_samples * res.label_length
        print(f'| {res.dataset:<{w}} | {res.num_samples:>9} | {res.accuracy:>8.2f} | {res.ned:>7.2f} '
              f'| {res.confidence:>10.2f} | {res.label_length:>12.2f} |', file=file)
    c.accuracy /= c.num_samples
    c.ned /= c.num_samples
    c.confidence /= c.num_samples
    c.label_length /= c.num_samples
    print('|-{:-<{w}}-|-----------|----------|---------|------------|--------------|'.format('----', w=w), file=file)
    print(f'| {c.dataset:<{w}} | {c.num_samples:>9} | {c.accuracy:>8.2f} | {c.ned:>7.2f} '
          f'| {c.confidence:>10.2f} | {c.label_length:>12.2f} |', file=file)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--data_root', default='/home/test13/yxm/data/STR/test')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--ost', action='store_true', default=False, help='Evaluate on OST datasets')
    parser.add_argument('--cased', action='store_true', default=False, help='Cased comparison')
    parser.add_argument('--punctuation', action='store_true', default=False, help='Check punctuation')
    parser.add_argument('--new', action='store_true', default=False, help='Evaluate on new benchmark datasets')
    parser.add_argument('--union', action='store_true', default=False, help='Evaluate on new benchmark datasets')
    parser.add_argument('--rotation', type=int, default=0, help='Angle of rotation (counter clockwise) in degrees.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--store_result', action='store_true', default=False, help='whether to store the results')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)

    charset_test = string.digits + string.ascii_lowercase
    if args.cased:
        charset_test += string.ascii_uppercase
    if args.punctuation:
        charset_test += "!\"#$%&'()*+,-./:<=>?@[\\]_£€¥×"
    kwargs.update({'charset_test': charset_test})
    print(f'Additional keyword arguments: {kwargs}')

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval()
    input=torch.randn(1, 3, 32, 128)
    flops, params = profile(model, inputs=(input, ))
    # flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
    
    model = model.to(args.device)
    hp = model.hparams
    datamodule = SceneTextDataModule(args.data_root, '_unused_', hp.img_size, hp.max_label_length, hp.charset_train,
                                     hp.charset_test, args.batch_size, args.num_workers, False, rotation=args.rotation)

    if args.ost:
        test_set = SceneTextDataModule.TEST_OST
    else:
        # test_set = SceneTextDataModule.TEST_BENCHMARK_SUB + SceneTextDataModule.TEST_BENCHMARK
        test_set = SceneTextDataModule.TEST_BENCHMARK
        
    if args.new:
        test_set += SceneTextDataModule.TEST_NEW
        
    if args.union:
        test_set = SceneTextDataModule.TEST_UNION
    test_set = sorted(set(test_set))

    results = {}
    max_width = max(map(len, test_set))
    if args.store_result:
        output_dir = os.path.join('./visualize', 'mpstr_union')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = open(os.path.join(output_dir, 'result.txt'), 'w', encoding='utf-8')
        output_file.write('GT\tPredict\tPre_Len\n')
            
    correct_len = {3:0, 4:0, 5:0, 6:0, 7:0, 8:0}
    total_len = {3:0, 4:0, 5:0, 6:0, 7:0, 8:0}
    
    for name, dataloader in datamodule.test_dataloaders(test_set).items():
        idx = 0    
        total = 0
        correct = 0
        ned = 0
        confidence = 0
        label_length = 0
        
        for imgs, labels in tqdm(iter(dataloader), desc=f'{name:>{max_width}}'):
            # _ = model(imgs.to(model.device))
            res = model.test_step((imgs.to(model.device), labels), -1)['output']
            
            total += res.num_samples
            correct += res.correct
            ned += res.ned
            confidence += res.confidence
            label_length += res.label_length
            
            for i in range(len(res.pred_list)):
                cur_len = len(res.gt_list[i])
                if cur_len <= 3:
                    total_len[3] += 1
                    if res.pred_list[i] == res.gt_list[i]:
                        correct_len[3] += 1
                elif cur_len >= 8:
                    total_len[8] += 1
                    if res.pred_list[i] == res.gt_list[i]:
                        correct_len[8] += 1
                else:
                    total_len[cur_len] += 1
                    if res.pred_list[i] == res.gt_list[i]:
                        correct_len[cur_len] += 1
            
            if args.store_result:
                for i in range(len(res.pred_list)):
                    # print(res.gt_list[i] + '\t' + res.pred_list[i] + '\n')
                    output_file.write(res.gt_list[i] + '\t' + res.pred_list[i] + '\t' + str(res.length_list[i]) + '\n')
                    if res.gt_list[i] != res.pred_list[i]:
                        # output_file.write(res.gt_list[i] + '\t' + res.pred_list[i] + '\n')
                        img_name = name + '-' + str(idx) + '-' + res.gt_list[i] + '-' + res.pred_list[i] + '-' + str(res.length_list[i]) + '.png'
                        img_path = os.path.join(output_dir, img_name)
                        cur_image = imgs[i].to(torch.device('cpu'))
                        vutils.save_image(cur_image, img_path)
                    
                    idx += 1
          
        accuracy = 100 * correct / total
        mean_ned = 100 * (1 - ned / total)
        mean_conf = 100 * confidence / total
        mean_label_length = label_length / total
        results[name] = Result(name, total, accuracy, mean_ned, mean_conf, mean_label_length)
        
    if args.ost:
        result_groups = {'OST': SceneTextDataModule.TEST_OST}
    else:
        result_groups = {
            # 'Benchmark (Subset)': SceneTextDataModule.TEST_BENCHMARK_SUB,
            'Benchmark': SceneTextDataModule.TEST_BENCHMARK
        }
    if args.new:
        result_groups.update({'New': SceneTextDataModule.TEST_NEW})
        
    if args.union:
        result_groups = {'UNION': SceneTextDataModule.TEST_UNION,}
        
    with open(args.checkpoint + '.log.txt', 'w') as f:
        for out in [f, sys.stdout]:
            for group, subset in result_groups.items():
                print(f'{group} set:', file=out)
                print_results_table([results[s] for s in subset], out)
                print('\n', file=out)
             
    print('length num acc.')   
    for key in correct_len:
        cur_accuracy = 100 * correct_len[key] / total_len[key]
        print(key, cur_accuracy, total_len[key])
    
if __name__ == '__main__':
    main()
