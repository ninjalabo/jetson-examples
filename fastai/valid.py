from datetime import datetime
from fastai.vision.all import *

models_dir = Path.cwd() / 'models'
models = ["resnet18", "resnet34", "resnet50"]

path = untar_data(URLs.IMAGENETTE_320,data=Path.cwd()/'data')
dls = ImageDataLoaders.from_folder(path, valid='val', item_tfms=Resize(224), batch_tfms=Normalize.from_stats(*imagenet_stats),)
logpath = Path.cwd() / 'valid_output.log'

def file_logger(line):
	print(line)
	with open(logpath, 'a') as logfile:
		logfile.write(line + '\n')

def validate_model(model_name, filename=None, models_directory=models_dir):
	filename = filename or (model_name + ".pkl")
	learn = load_learner(Path(models_directory) / filename)
	
	print("Validation with Imagenette 320")
	start_time = datetime.now()
	loss, acc = learn.validate(dl=dls[1])
	end_time = datetime.now()

	print(f"  - Loss: {loss}")
	print(f"  - Accuracy: {acc}")
	print(f"  - Time: {end_time - start_time}")

	return loss, acc

def main():
	ready = False
	while not ready:
		model_name = input("Select saved model to validate: ")
		if model_name in models:
			ready = True
	validate_model(model_name)

if __name__ == "__main__":
	main()
