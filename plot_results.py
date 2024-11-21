import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot Test Accuracy vs Removal Epoch for Different Deficits')
    parser.add_argument('--log_dir', type=str, default='logs/', help='Directory containing CSV logs')
    parser.add_argument('--output', type=str, default='accuracy_plot.png', help='Output plot file name')
    return parser.parse_args()

def main():
    args = parse_arguments()
    log_dir = args.log_dir
    output_file = args.output

    # 定义缺陷类型
    deficits = ["blur", "vertical_flip", "label_permutation", "noise", "none"]

    # 初始化数据结构
    results = {deficit: {} for deficit in deficits if deficit != "none"}
    baseline = {}

    # 遍历日志文件
    for filename in os.listdir(log_dir):
        if not filename.endswith('.csv'):
            continue
        filepath = os.path.join(log_dir, filename)
        df = pd.read_csv(filepath)

        # 提取run_name
        run_name = filename.replace('.csv', '')
        if run_name.startswith("baseline"):
            # 处理基线
            # 假设基线的val_acc随epoch变化
            # 选择最后一个epoch的val_acc
            final_epoch = df['epoch'].max()
            final_acc = df[df['epoch'] == final_epoch]['val_acc'].values[0]
            baseline['val_acc'] = final_acc
        else:
            # 提取deficit类型和移除epoch
            parts = run_name.split('_de')
            if len(parts) != 2:
                continue
            deficit_type = parts[0]
            removal_epoch = int(parts[1])

            # 获取在移除epoch时的val_acc
            if removal_epoch in df['epoch'].values:
                acc = df[df['epoch'] == removal_epoch]['val_acc'].values[0]
                results[deficit_type][removal_epoch] = acc

    # 绘制图形
    plt.figure(figsize=(10, 6))

    # 绘制每种缺陷的曲线
    for deficit, epochs_acc in results.items():
        epochs = sorted(epochs_acc.keys())
        accs = [epochs_acc[epoch] for epoch in epochs]
        plt.plot(epochs, accs, marker='o', label=deficit)

    # 绘制基线
    if 'val_acc' in baseline:
        plt.axhline(y=baseline['val_acc'], color='k', linestyle='--', label='baseline')

    plt.xlabel('Removal Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy at Removal Epoch for Different Deficits')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.show()

if __name__ == "__main__":
    main()
