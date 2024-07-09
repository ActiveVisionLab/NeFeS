# !/bin/bash

### Train NeFeS stage1 photometric loss only ###
# python run_nefes.py --config config/7Scenes/dfnet/config_stairs_stage1.txt

### Train NeFeS stage2 photometric+featuremetric loss ###
# python run_nefes.py --config config/7Scenes/dfnet/config_stairs_stage2.txt

### Train NeFeS stage1 photometric loss only ###
python run_nefes.py --config config/Cambridge/dfnet/config_shop_stage1.txt

### Train NeFeS stage2 photometric+featuremetric loss ###
python run_nefes.py --config config/Cambridge/dfnet/config_shop_stage2.txt