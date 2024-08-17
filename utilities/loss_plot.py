import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Loss plot')
parser.add_argument('log', type=str)


def extract_items(log, mode):
    log = log.replace('      ', '    ')
    log_items = log.split('	')
    epoch = log_items[0]
    epoch = epoch.replace('Epoch: [', '')
    epoch = int(epoch[:epoch.find('][')])
    loss = float(log_items[3].split(' ')[1])
    if mode == 'all':
        acc1 = log_items[4].replace('Acc@1', '').lstrip(' ').split(' ')[0]
        acc1 = float(acc1)
        acc5 = log_items[5].replace('Acc@5', '').lstrip(' ').split(' ')[0]
        acc5 = float(acc5)
        return epoch, loss, acc1, acc5
    elif mode == 'loss':
        return epoch, loss


def extract_epoch_loss_acc1_acc5(log: str, mode='all'):
    if not log.startswith('Epoch: ['):
        log = log.lstrip('\x00')
    if 130 < len(log) < 140 and mode == 'all':
        epoch, loss, acc1, acc5 = extract_items(log, mode)
        return True, epoch, loss, acc1, acc5
    elif mode == 'loss':
        epoch, loss = extract_items(log, mode)
        return True, epoch, loss
    else:
        if mode == 'all':
            return False, 0, 0, 0, 0
        elif mode == 'loss':
            return False, 0, 0


def read_train_log(train_log_path, mode='all'):
    epochs = []
    loss_values = []
    acc1_values = []
    acc5_values = []
    if mode == 'all':
        with open(train_log_path, 'r') as f:
            epoch_previous = -1
            count = 0
            loss_sum = 0
            acc1_sum = 0
            acc5_sum = 0
            while line := f.readline():
                valid, epoch, loss, acc1, acc5 = extract_epoch_loss_acc1_acc5(line, mode)
                if valid:
                    count += 1
                    loss_sum += loss
                    acc1_sum += acc1
                    acc5_sum += acc5
                    if epoch != epoch_previous:
                        epochs.append(epoch)
                        loss_values.append(loss_sum / count)
                        acc1_values.append(acc1_sum / count)
                        acc5_values.append(acc5_sum / count)
                        epoch_previous = epoch
                        count = 0
                        loss_sum = 0
                        acc1_sum = 0
                        acc5_sum = 0
        return epochs, loss_values, acc1_values, acc5_values
    elif mode == 'loss':
        with open(train_log_path, 'r') as f:
            epoch_previous = -1
            count = 0
            loss_sum = 0
            while line := f.readline():
                valid, epoch, loss = extract_epoch_loss_acc1_acc5(line, mode)
                if valid:
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
    mode = 'all'  # to plot either all loss, acc1, and acc5 or only loss (mode = 'loss')
    if mode == 'all':
        epoch_arr, loss_arr, acc1_arr, acc5_arr = read_train_log(args.log, mode)
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(np.array(epoch_arr), np.array(loss_arr), label='loss', color='r')
        ax2.plot(np.array(epoch_arr), np.array(acc1_arr), label='acc1', color='g', linestyle='--')
        ax2.plot(np.array(epoch_arr), np.array(acc5_arr), label='acc5', color='y', linestyle='--')
        fig.legend(loc='upper right')
        plt.show()
    elif mode == 'loss':
        epoch_arr, loss_arr = read_train_log(args.log, mode)
        fig, ax1 = plt.subplots()
        ax1.plot(np.array(epoch_arr), np.array(loss_arr), label='loss', color='r')
        fig.legend(loc='upper right')
        plt.show()
