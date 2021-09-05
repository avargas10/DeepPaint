# Move data into training and validation directories
import os
os.makedirs('images/train/class/', exist_ok=True) # 10,000 images
os.makedirs('images/val/class/', exist_ok=True)   #  1,000 images
for i, file in enumerate(os.listdir('testSet_resize')):
  if i < 1000: # first 1000 will be val
    os.rename('testSet_resize/' + file, 'images/val/class/' + file)
  elif (1000 <= i and i < 10000): # others will be val
    os.rename('testSet_resize/' + file, 'images/train/class/' + file)
    
# Make sure the images are there
from IPython.display import Image, display
display(Image(filename='images/val/class/84b3ccd8209a4db1835988d28adfed4c.jpg'))