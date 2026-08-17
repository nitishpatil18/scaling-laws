import sys
sys.path.insert(0, 'src')
from train import train
train('64M', tokens_per_param=20, batch_size=32, log_every=200)
