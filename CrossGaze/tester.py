import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from models import *
from utils import *
from helpers import *
from evaluators import *
import numpy as np
import random
import pandas as pd
# Arguments
args = define_args()
config = vars(args)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# Hyperparameters
workers = args.num_workers
epochs = args.epochs
batch_size = args.batch_size
lr = args.lr

run_name = args.run_name

class PartEvaluatorTest():
	def __init__(self, args, val_dataset, val_loader, device, criterion, metric, save_file):
		self.args = args
		self.val_loader = val_loader
		self.device = device
		self.criterion = criterion
		self.metric = metric
		self.val_dataset = val_dataset

		self.results = []
		self.filenames = []
		self.demographic_labels = []

		self.save_file = save_file
	
	def eval(self, model):
		model.eval()
		total_val_loss = 0
		avg_error = 0
		with torch.no_grad():
			for i, data in enumerate(self.val_loader):
				images = data['img']
				labels = data['label3d'].to(self.device)

				face_images = images[0].to(self.device)
				left_eye_images = images[1].to(self.device)
				right_eye_images = images[2].to(self.device)
				pred = model(face_images, left_eye_images, right_eye_images)
				loss = self.criterion(pred, labels)

				total_val_loss += loss.item() * 100

				distance = self.metric(pred, labels)
				for i in range(len(pred)):
					norm = torch.linalg.norm(pred[i])				
					normalized_pred = pred[i]/norm
					error = self.metric(normalized_pred.unsqueeze(0), labels[i].unsqueeze(0))
					self.results.append(error.cpu().detach().numpy())
					self.filenames.append(data['name'][i])
					self.demographic_labels.append(data['demographic_label'][i])
				avg_error += distance.item()

		angular_error = avg_error / len(self.val_dataset)
		val_loss = total_val_loss / len(self.val_dataset)
		data = {'errors': self.results, 'filenames': self.filenames, 'DemographicLabel': self.demographic_labels}
		dataframe = pd.DataFrame.from_dict(data)
		dataframe.to_csv(self.save_file)

		return angular_error, val_loss

def main():
	# For logging and saving checkpoints
	os.makedirs('weights', exist_ok=True)
	os.makedirs('results', exist_ok=True)
	f = open('results/' + run_name + '.txt', 'w')

	# Augmentation
	train_transforms, test_transforms = get_transforms(args)

	# Loaders
	train_dataset, train_loader, val_dataset, val_loader = get_loaders(args, train_transforms, test_transforms, 
																	   label_name=args.label_name, test_label_name=args.test_label_name, use_reweight=False, 
																	   use_oversample=False, use_class_weights=False)

	# Model
	model = choose_model(args)

	print(args.model + ": Number of Parameters - " + str(count_parameters(model)))
	if args.checkpoint:
		model = load_checkpoint(model, args.checkpoint)

	model= nn.DataParallel(model)
	model.to(device)

	# Evaluation Metric and Loss
	metric = AngularDistance()

	if args.loss == 'mse':
		criterion = nn.MSELoss().cuda()
	elif args.loss == 'angular':
		criterion = CosineSimilarityLoss().cuda()
	elif args.loss == 'mse-angular':
		criterion = MSEAngularLoss(args.loss_alpha).cuda()

	# Optimizer and Scheduler
	optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
	scheduler = get_scheduler(args, optimizer, train_loader)

	evaluator = PartEvaluatorTest(args, val_dataset, val_loader, device, criterion, metric, save_file=args.save_file)
	# Training Loop
	min_angular_error = 100

	angular_error, val_loss = evaluator.eval(model)
	print("VAL LOSS: " + str(val_loss))
	print("ANGULAR ERROR: " + str(angular_error) + " Current Best: " + str(min_angular_error) + "\n")

	print("-" * 50)
	print("Finished Training")
	print("Best Angular Error: " + str(min_angular_error))
	print("-" * 50, "\n")

if __name__ == '__main__':
	main()