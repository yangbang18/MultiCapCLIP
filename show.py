import argparse
import os
import glob
import numpy as np
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='output')
parser.add_argument('--fn', type=str, default='log.txt')
parser.add_argument('--specific', type=str)
parser.add_argument('--monitor', type=str, default='CIDEr')
parser.add_argument('--metrics', type=str, nargs='+', default=['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE'])
parser.add_argument('--ndigits', type=int, default=1)
parser.add_argument('--csv_path', type=str, default='.')
parser.add_argument('--csv_fn', type=str, default='merge.csv')

parser.add_argument('-ss', '--specific_subset', type=str, choices=['0.01%', '0.1%', '1%', '10%'])
parser.add_argument('-sk', '--specific_key', type=str)
parser.add_argument('--epoch', type=int, default=10)

parser.add_argument('--mode', type=str, default='test', choices=['test', 'val'])
args = parser.parse_args()

results = {}
method2results = defaultdict(dict)

def loop(path):
    if os.path.isfile(path):
        if args.fn in path and 'subset' in path:
            data = open(path, 'r').read().strip().split('\n')
            data = [line for line in data if 'best' not in line and 'val' in line]

            method = '/'.join(path.split('/')[-4:-2])
            method = method.replace(',', '&')
            folder = os.path.basename(os.path.dirname(path))

            subset = folder.split('_')[0]
            if subset not in method2results[method]:
                method2results[method][subset] = []


            if args.specific and args.specific not in folder:
                print('### Skip', folder)
            else:
                best_score = 0
                best_line = None
                for line in data:
                    line = eval(line)
                    if line['epoch'] >= args.epoch:
                        continue
                    d = line[f'val_{args.monitor}']
                    if d > best_score:
                        best_score = d
                        best_line = line

                if best_line is not None:
                    results[folder] = best_line
                    method2results[method][subset].append(best_line)
                else:
                    print('### Failed to process', folder, method, data)
    else:
        for p in glob.glob(os.path.join(path, '*')):
            loop(p)

loop(args.root)


def convert(num):
    num *= 100
    num = round(num, args.ndigits)
    return num

def merge(lines):
    n = len(lines)
    mean = {}
    std = {}
    for k in args.metrics:
        key = f'{args.mode}_{k}'
        tmp = []
        for i in range(n):
            tmp.append(lines[i].get(key, 0))
        
        mean[key] = np.array(tmp).mean() if len(tmp) else 0
        std[key] = np.array(tmp).std() if len(tmp) else 0
    
    return mean, std, n

max_length = max([len(k) for k in method2results.keys()])

# header
f = open(args.csv_fn, 'w')
keys = sorted(list(method2results.keys()))
os.makedirs(args.csv_path, exist_ok=True)
with open(os.path.join(args.csv_path, args.csv_fn), 'w') as f:
    
    print(f'%{max_length}s' % 'method', 'subset', '\t'.join(['%10s' % metric for metric in args.metrics]), '\t%s' % 'n')
    f.write(','.join(['method', 'subset'] + args.metrics + ['n']) + '\n')

    for k in keys:
        if args.specific_key and args.specific_key not in k:
            continue
        v = method2results[k]
        for subset, vv in v.items():
            if args.specific_subset and subset != args.specific_subset:
                continue

            mean, std, n = merge(vv)
            mean = [convert(mean[f'{args.mode}_{metric}']) for metric in args.metrics]
            std = [convert(std[f'{args.mode}_{metric}']) for metric in args.metrics]
            nums_to_show = ['%4.1f (%3.2f)' % (m, s) for m, s in zip(mean, std)]
            nums_to_show.append('%d' % n)

            print(f'%{max_length}s %6s'%(k, subset), '\t'.join(nums_to_show))
            f.write(','.join([k, subset] + nums_to_show) + '\n')
