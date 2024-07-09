# !/bin/bash

################################################### Evaluate paper Exp. result ########################################################################
### 7Scenes sfm apr refinement
python eval.py --config config/7Scenes/dfnet/config_heads_DFM.txt 
python eval.py --config config/7Scenes/dfnet/config_fire_DFM.txt 
python eval.py --config config/7Scenes/dfnet/config_chess_DFM.txt 
python eval.py --config config/7Scenes/dfnet/config_office_DFM.txt 
python eval.py --config config/7Scenes/dfnet/config_pumpkin_DFM.txt 
python eval.py --config config/7Scenes/dfnet/config_kitchen_DFM.txt 
python eval.py --config config/7Scenes/dfnet/config_stairs_DFM.txt 

### Cambridge, dataloader is slower to initialize due to preload policy.
python eval.py --config config/Cambridge/dfnet/config_shop_DFM.txt
python eval.py --config config/Cambridge/dfnet/config_kings_DFM.txt
python eval.py --config config/Cambridge/dfnet/config_hospital_DFM.txt
python eval.py --config config/Cambridge/dfnet/config_church_DFM.txt
