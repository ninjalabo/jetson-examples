import os
import pandas as pd
from fastai.vision.all import *

models = {
	'resnet18': resnet18,
	'resnet34': resnet34,
	'resnet50': resnet50,
}
save_dir = Path.cwd()/'models'
path = untar_data(URLs.IMAGENETTE_320,data=Path.cwd()/'data')
dls = ImageDataLoaders.from_folder(path, valid='val', item_tfms=Resize(224), batch_tfms=Normalize.from_stats(*imagenet_stats),)
logpath = Path.cwd()/'train_output.log'

def file_logger(line):
	print(line)
	with open(logpath, 'a') as file:
		file.write(line + '\n')

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
	learn = train(model_name, epochs)
	save_trained_learner(learn, model_name + '.pkl')

if __name__ == "__main__":
	main()
