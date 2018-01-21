import Augmentor
import numpy as np

for i in np.arange(0,3): 
	p=Augmentor.Pipeline("/home/luffyfjx/workspace/TDCV_RF/TDCV_RF/data/task3/train/0"+str(i))
	p.rotate90(probability=0.8)
	p.rotate270(probability=0.8)
	p.flip_left_right(probability=1)
	p.flip_top_bottom(probability=1)
	p.skew(0.8)
	p.shear(0.8, 15,15)
	# p.crop_random(probability=0.8,percentage_area=0.9)
	p.sample(1000)

p=Augmentor.Pipeline("/home/luffyfjx/workspace/TDCV_RF/TDCV_RF/data/task3/train/03")
p.rotate90(probability=0.5)
p.rotate270(probability=0.5)
p.flip_left_right(probability=0.5)
p.flip_top_bottom(probability=0.5)
p.skew(probability=0.8)
# p.crop_random(probability=1,percentage_area=0.9)
p.sample(1000)