import matplotlib.pyplot as plt

metrics = ['loss', 'accuracy', 'recall_m', 'precision_m', 'f1_m', 'mean_io_u']


def plot_record(record):

    for metric in metrics:
        fig = plt.figure(figsize=(10, 6))
        plt.plot(record[metrics], label='training')
        plt.plot(record[f'val_{metrics}'], label='validation')
        plt.ylabel(metrics, fontsize=20)
        plt.xlabel('Epoch', fontsize=20)
        plt.legend(fontsize=20)
        plt.show()
