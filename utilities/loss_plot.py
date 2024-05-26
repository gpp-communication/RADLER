import argparse
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Loss plot')
parser.add_argument('log', type=str)


def extract_epoch_loss(log: str):
    if not log.startswith('Epoch: ['):
        log = log.lstrip('\x00')
    log = log.replace('      ', '    ')
    log_items = log.split('	')
    epoch = log_items[0]
    epoch = epoch.replace('Epoch: [', '')
    epoch = int(epoch[:epoch.find('][')])
    loss = float(log_items[3].split(' ')[1])
    return epoch, loss


def read_train_log(train_log_path):
    epochs = []
    loss_values = []
    with open(train_log_path, 'r') as f:
        epoch_previous = -1
        count = 0
        loss_sum = 0
        while line := f.readline():
            epoch, loss = extract_epoch_loss(line)
            count += 1
            loss_sum += loss
            if epoch != epoch_previous:
                epochs.append(epoch)
                loss_values.append(loss_sum / count)
                epoch_previous = epoch
                count = 0
                loss_sum = 0
    return epochs, loss_values


if __name__ == '__main__':
    args = parser.parse_args()
    epoch_arr, loss_arr = read_train_log(args.log)
    plt.plot(np.array(epoch_arr), np.array(loss_arr))
    plt.show()
