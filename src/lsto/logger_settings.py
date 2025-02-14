import os
import json
from logging import config

file_dir = os.path.dirname(__file__)

with open(file_dir+'/log/log_config.json', 'r') as f:
  log_conf = json.load(f)

config.dictConfig(log_conf)
