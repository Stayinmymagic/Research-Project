#!/usr/local/bin/python

import yaml
import itertools
import os
import logging

class PathParser:

    def __init__(self, config_path):
        # self.root = '/home/lab05/Desktop/Research_code/'
        # self.data = os.path.join(self.root, config_path['data'])
        self.data = config_path['data']
        self.log =  config_path['log']
        self.target = os.path.join(self.data, config_path['target'])
        self.fileName = os.path.join(self.data, config_path['text'])


config_fp = os.path.join(os.path.dirname(__file__), 'config.yml')
with open(config_fp, 'r') as file:
    config = yaml.full_load(file)
config_model = config['model']
topics = config['topics']
path_parser = PathParser(config_path=config['paths'])

#logger
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)
log_fp = os.path.join(path_parser.log, '{0}.log'.format('model'))
file_handler = logging.FileHandler(log_fp)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


