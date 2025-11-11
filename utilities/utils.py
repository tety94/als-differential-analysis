import logging
import os
import matplotlib.pyplot as plt

def init_logging(log_path):
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s - %(message)s')
    return logging.info

def save_plot(fig, path):
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
