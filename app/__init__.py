#-*- coding:utf-8-*-
from flask import Flask

app = Flask(__name__)
app.config.from_object('config_lv')

import sys
sys.path.append("/home/season/season/Study36/")
sys.path.append("/home/season/season/Study36/mask_rcnn/")

from app import views
