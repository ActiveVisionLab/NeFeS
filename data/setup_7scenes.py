# modified from https://github.com/vislearn/dsacstar/blob/master/datasets/setup_7scenes.py
import os

# name of the folder where we download the original 7scenes dataset to
# we restructure the dataset by creating symbolic links to that folder
src_src_folder='deepslam_data'
src_folder = '7Scenes'
colmap_poses = '7Scenes_colmap_poses'
# focallength = 525.0

def mkdir(directory):
	"""Checks whether the directory exists and creates it if necessacy."""
	if not os.path.exists(directory):
		os.makedirs(directory)

# download the original 7 scenes dataset for poses and images
mkdir(src_src_folder)
mkdir(src_src_folder+'/'+src_folder)
os.chdir(src_src_folder+'/'+src_folder)

for ds in ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']:
	print("=== Downloading 7scenes Data:", ds, "===============================")

	os.system('wget http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/' + ds + '.zip')
	os.system('unzip ' + ds + '.zip')
	os.system('rm ' + ds + '.zip')
	
	sequences = os.listdir(ds)

	for file in sequences:
		if file.endswith('.zip'):

			print("Unpacking", file)
			os.system('unzip ' + ds + '/' + file + ' -d ' + ds)
			os.system('rm ' + ds + '/' + file)

	print("Copying colmap pose files...")
	os.system(f'cp ../../7Scenes_colmap_poses/{ds}/*.txt {ds}/')
