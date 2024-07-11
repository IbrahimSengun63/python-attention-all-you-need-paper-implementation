import torch
import warnings
from train_model import train_model
from config import get_config
# use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.device_count())
print(device)



warnings.filterwarnings('ignore')
config = get_config()
train_model(config)
