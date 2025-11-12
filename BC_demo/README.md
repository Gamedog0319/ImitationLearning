# Behavior Cloning (BC) Demo for Malmo
**Authors:** Zhengyao Li, Joy Zhuo  
**Date:** October 27 2025  

## Features
This script supports three main actions:
1. **Generate Fake Dataset:** create synthetic arrow images and labels (LEFT, RIGHT, GO, BACK)
2. **Train CNN Model:** learn to predict actions from those images
3. **Run Inference:** test the trained model on an example image

## Requirements
You only need **Python 3.9+** and **PyTorch**.

If you use **Conda**, setup is simple:

conda create -n bc python=3.10  
conda activate bc  
pip install torch torchvision pillow numpy  

## How to run
**1. Generate a fake dataset (3000 samples)**
python bc.py --make-fake 3000

**2. Train the CNN behavior cloning model**
python bc.py --train

**3. Test a trained model on one sample image**
python bc.py --predict

**When training completes, the model will be saved as:**
malmo_bc_cnn.pth
