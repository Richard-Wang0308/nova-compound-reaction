# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv

load_dotenv()

class RuntimeConfig:
    PSICHIC_PATH = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(PSICHIC_PATH, 'trained_weights', 'TREAT1')
    BATCH_SIZE = 128
    
    @property
    def DEVICE(self):
        """Read device from environment variable dynamically (not at import time)."""
        device = os.environ.get("DEVICE_OVERRIDE")
        return "cpu" if device == "cpu" else "cuda:0"
    
