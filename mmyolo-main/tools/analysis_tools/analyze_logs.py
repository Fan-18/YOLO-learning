import argparse
import json
import matplotlib.pyplot as plt
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Plot train log')
    parser.add_argument('json_logs', help='path of train log file', nargs='+')
    parser.add_argument('--keys', help='the keywords to plot', nargs='+', default=['loss'])
    parser.add_argument('--legend', help='the legend of each plot', nargs='+')
    parser.add_argument('--out', help='output path of figure', default='curve.png')
    return parser.parse_args()

def plot_curve(log_dicts, args):
    plt.figure(figsize=(10, 5))
    for i, log_dict in enumerate(log_dicts):
        for key in args.keys:
            if key not in log_dict:
                continue
            xs = log_dict['step']
            ys = log_dict[key]
            label = args.legend[i] if args.legend else key
            plt.plot(xs, ys, label=label)
    plt.xlabel('step')
    plt.grid(True)
    plt.legend()
    plt.savefig(args.out)
    print(f'成功保存图表到: {args.out}')

def load_json_log(json_log):
    log_dict = dict()
    with open(json_log, 'r') as f:
        for line in f:
            log_item = json.loads(line.strip())
            if 'step' not in log_item:
                continue
            step = log_item['step']
            for key, val in log_item.items():
                if key not in log_dict:
                    log_dict[key] = []
                    log_dict['step'] = []
                if key in log_item:
                    log_dict[key].append(log_item[key])
                    if 'step' not in log_dict: log_dict['step'].append(step)
    # 对齐 step
    min_len = min(len(v) for v in log_dict.values())
    for key in log_dict:
        log_dict[key] = log_dict[key][:min_len]
    return log_dict

if __name__ == '__main__':
    args = parse_args()
    log_dicts = [load_json_log(json_log) for json_log in args.json_logs]
    plot_curve(log_dicts, args)
