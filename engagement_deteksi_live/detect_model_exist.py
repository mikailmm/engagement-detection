import cv2
import numpy as np
import joblib

MODEL_PATH = 'boredom-model2-randomforest'

try:
    model = joblib.load(MODEL_PATH)
    print(f"Model at '{MODEL_PATH}' loaded")
except FileNotFoundError:
    print(f"Error: Model not found at '{MODEL_PATH}'")
    exit()
