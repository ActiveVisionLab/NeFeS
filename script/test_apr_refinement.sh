# !/bin/bash

## NeFeS + APR refinement ###
 python test_refinement.py --config config/7Scenes/dfnet/config_stairs_DFM.txt
#  python test_refinement.py --config config/7Scenes/dfnet/config_heads_DFM.txt
#  python test_refinement.py --config config/7Scenes/dfnet/config_chess_DFM.txt
#  python test_refinement.py --config config/7Scenes/dfnet/config_fire_DFM.txt
#  python test_refinement.py --config config/7Scenes/dfnet/config_kitchen_DFM.txt
#  python test_refinement.py --config config/7Scenes/dfnet/config_pumpkin_DFM.txt
#  python test_refinement.py --config config/7Scenes/dfnet/config_office_DFM.txt

#  python test_refinement.py --config config/Cambridge/dfnet/config_shop_DFM.txt
#  python test_refinement.py --config config/Cambridge/dfnet/config_hospital_DFM.txt
#  python test_refinement.py --config config/Cambridge/dfnet/config_kings_DFM.txt
#  python test_refinement.py --config config/Cambridge/dfnet/config_church_DFM.txt

## NeFeS + Pose refinement (Table 5.) ###
# python test_refinement.py --config config/7Scenes/dfnet/config_stairs_DFM.txt --pose_only 3 --lr_r 0.0087 --lr_t 0.01
#  python test_refinement.py --config config/7Scenes/dfnet/config_heads_DFM.txt --pose_only 3 --lr_r 0.0087 --lr_t 0.01
#  python test_refinement.py --config config/7Scenes/dfnet/config_chess_DFM.txt --pose_only 3 --lr_r 0.0087 --lr_t 0.01
#  python test_refinement.py --config config/7Scenes/dfnet/config_fire_DFM.txt --pose_only 3 --lr_r 0.0087 --lr_t 0.01
#  python test_refinement.py --config config/7Scenes/dfnet/config_kitchen_DFM.txt --pose_only 3 --lr_r 0.0087 --lr_t 0.01
#  python test_refinement.py --config config/7Scenes/dfnet/config_pumpkin_DFM.txt --pose_only 3 --lr_r 0.0087 --lr_t 0.01
#  python test_refinement.py --config config/7Scenes/dfnet/config_office_DFM.txt --pose_only 3 --lr_r 0.0087 --lr_t 0.01

#  python test_refinement.py --config config/Cambridge/dfnet/config_shop_DFM.txt --pose_only 3
#  python test_refinement.py --config config/Cambridge/dfnet/config_hospital_DFM.txt --pose_only 3
#  python test_refinement.py --config config/Cambridge/dfnet/config_kings_DFM.txt --pose_only 3
#  python test_refinement.py --config config/Cambridge/dfnet/config_church_DFM.txt --pose_only 3