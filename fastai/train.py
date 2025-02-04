import os
import pandas as pd
from datetime import datetime
from fastai.vision.all import *

models = {
	'resnet18': resnet18,
	'resnet34': resnet34,
	'resnet50': resnet50,
}
save_dir = Path.cwd()/'models'
path = untar_data(URLs.IMAGENETTE_320,data=Path.cwd()/'data')
dls = ImageDataLoaders.from_folder(path, valid='val', item_tfms=Resize(224), batch_tfms=Normalize.from_stats(*imagenet_stats),)
logpath = None

def setup_logfile(model_name):
	global logpath
	timestamp = str(int(datetime.now().timestamp()))
	os.makedirs(Path.cwd() / 'logs', exist_ok=True)
	logpath = Path.cwd() / 'logs' / f'train_{model_name}_{timestamp}.txt'
	with open(logpath, 'w') as logfile:
		logfile.write(f'Training model {model_name} with Imagenette \n')
		cols = ['epoch', 'train_loss', 'valid_loss', 'accuracy', 'time']
		logfile.write('\n')
		logfile.write('\t'.join(cols) + '\n')

def file_logger(line):
	strline = [ round(ele, 9) if isinstance(ele, (int, float)) else ele for ele in line ]
	strline = [ str(ele) for ele in strline ]
	strline = '\t'.join(strline)
	print(strline)
	with open(logpath, 'a') as file:
		file.write(strline + '\n')

def train(model_name: str, epochs: int = 1):
	if model_name not in models:
		raise ValueError(f"Model name {model_name} is not supported. Choose from: {list(models.keys())}")
	model = models[model_name]
	learn = vision_learner(dls, model, metrics=accuracy, pretrained=True)
	learn.recorder.logger = file_logger
	learn.fine_tune(epochs)
	
	return learn

def save_trained_learner(learn, filename = "model.pkl", save_directory = save_dir):
	os.makedirs(save_directory, exist_ok=True)
	learn.path = Path(save_directory)
	learn.export(fname=filename)

def main():
	ready = False
	while not ready:
		model_name = input('Select a model to train (resnet18/34/50): ')
		epochs = input('Select the number of epochs: ')
		try:
			epochs = int(epochs)
		except:
			continue
		if model_name in models:
			ready = True
	setup_logfile(model_name)
	learn = train(model_name, epochs)
	save_trained_learner(learn, model_name + '.pkl')

if __name__ == "__main__":
	main()
