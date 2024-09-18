from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import os
import cv2
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

driver = webdriver.Chrome()
LINK = "https://jstris.jezevec10.com/join/RLF0C8"
driver.get(LINK)

s_folder = r"C:\Users\ryand\Learning\GPT Calculator\screenshots"

def convert_to_grid(img_path):
    left, top, right, bottom = 232, 154, 594, 875

    img = Image.open(img_path)

    game_area = img.crop((left, top, right, bottom))
    game_area.save("cropped_game_area.png")

    bgr_game = cv2.imread(r"cropped_game_area.png")
    hsv_game = cv2.cvtColor(bgr_game, cv2.COLOR_BGR2HSV)


    t1, t2 = 10, 40
    color_ranges = {
        "red": ((174 - t1, 235 - t2, 208 - t2), (174 + t1, 235 + t2, 208 + t2)),
        "light blue": ((101 - t1, 187 - t2, 214 - t2), (101 + t1, 187 + t2, 214 + t2)),
        "dark blue": ((118 - t1, 185 - t2, 197 - t2), (118 + t1, 185 + t2, 197 + t2)),
        "yellow": ((21 - t1, 235 - t2, 221 - t2), (21 + t1, 235 + t2, 221 + t2)),
        "orange": ((11 - t1, 240 - t2, 220 - t2), (11 + t1, 240 + t2, 220 + t2)),
        "green": ((45 - t1, 258 - t2, 177 - t2), (45 + t1, 258 + t2, 177 + t2)),
        "pink": ((158 - t1, 195 - t2, 171 - t2), (158 + t1, 195 + t2, 171 + t2)),
    }

    combined_mask = np.zeros(hsv_game.shape[:2], dtype = np.uint8)
    for color, (lower_hsv, upper_hsv) in color_ranges.items():
        mask = cv2.inRange(hsv_game, np.array(lower_hsv), np.array(upper_hsv))
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    grid = cv2.resize(combined_mask, (10, 20), interpolation = cv2.INTER_NEAREST)
    binary_grid = (grid > 0).astype(int)
    return binary_grid