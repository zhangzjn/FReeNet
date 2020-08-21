import random
from torch.utils.data import dataset
import torch
import os
import copy
import glob


class Landmarks_RaFD(dataset.Dataset):
	def __init__(self):
		self.calms = list()
		self.shapes = list()
		self.name2shape = dict()
		landmark_paths = glob.glob('dataset_RaFD/landmark_crop_*.txt')
		landmark_paths.sort()

		for landmark_path in landmark_paths:
			labs = open(landmark_path, 'r').readlines()
			for i, lab in enumerate(labs):
				name, shape = self.parse_lab(lab)
				# if name.find('neutral_frontal') != -1 and name.find('Rafd090') != -1:
				if name.find('neutral_frontal') != -1:
					self.calms.append([name, shape])
				self.shapes.append([name, shape])
				self.name2shape[name] = shape
		self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}
		self.uids = self.unique_ids

	def parse_lab(self, lab):
		l = lab.strip().split()
		name = l[0]
		w_ori, h_ori = [int(_) for _ in l[1].split('-')]
		shape = []
		for l_ in l[2:]:
			w, h = [float(_) for _ in l_.split('-')]
			shape.extend([w, h])
		return name, shape

	def __getitem__(self, index):
		A_name, A_shape = self.shapes[index]
		B_name_calm, B_shape_calm = random.sample(self.calms, 1)[0]
		A_emotion = self.emotion(A_name)
		B_emotion_calm = self.emotion(B_name_calm)
		B_name = B_name_calm.replace(B_emotion_calm, A_emotion)
		B_shape = self.name2shape[B_name]
		A_name_calm = A_name.replace(A_emotion, B_emotion_calm)
		A_shape_calm = self.name2shape[A_name_calm]

		R_name, _ = random.sample(self.shapes, 1)[0]
		R_angle, R_emotion, R_direction = self.information(R_name)
		B_angle, B_emotion, B_direction = self.information(B_name)
		B1_name = B_name.replace(B_angle, R_angle).replace(B_emotion, R_emotion).replace(B_direction, R_direction)
		B1_shape = self.name2shape[B1_name]

		A_shape = torch.Tensor(A_shape)
		B_shape = torch.Tensor(B_shape)
		B1_shape = torch.Tensor(B1_shape)
		A_shape_calm = torch.Tensor(A_shape_calm)
		B_shape_calm = torch.Tensor(B_shape_calm)

		return [A_shape, B_shape, B1_shape, A_shape_calm, B_shape_calm]

	def __len__(self):
		return len(self.shapes)

	@staticmethod
	def emotion(img_path):
		return img_path.split('_')[-2]

	@staticmethod
	def information(img_path):
		return img_path.split('_')[0], img_path.split('_')[-2], img_path.split('_')[-1]

	@staticmethod
	def id(img_path):
		return img_path.split('_')[1]

	@property
	def ids(self):
		return [self.id(os.path.basename(label[0])) for label in self.shapes]

	@property
	def unique_ids(self):
		return sorted(set(self.ids))


class Landmarks_PIE(dataset.Dataset):
	def __init__(self):
		self.ids = set()
		self.id2names = dict()
		self.name2shape = dict()

		land_path = '/media/datasets/zhangzjn/Multi_PIE_Part/landmark_crop.txt'
		labs = open(land_path).readlines()
		for l in labs:
			l = l.strip().split()
			name = l[0]
			w_ori, h_ori = [int(_) for _ in l[1].split('-')]
			shape = []
			for l_ in l[2:108]:
				w, h = [float(_) for _ in l_.split('-')]
				shape.extend([w, h])
			pose = l[108:]

			i_id, i_expression, i_angle, _ = name.split('.')[0].split('_')
			self.ids.add(i_id)
			if i_id not in self.id2names.keys():
				self.id2names[i_id] = [name]
			else:
				self.id2names[i_id].append(name)
			self.name2shape[name] = shape
		self.ids = list(self.ids)

	def __getitem__(self, index):
		A_id = self.ids[index]
		B_id = random.sample(self.ids, 1)[0]
		A_name = random.sample(self.id2names[A_id], 1)[0]
		B_name = random.sample(self.id2names[B_id], 1)[0]
		B_name = B_name[0:4] + A_name[4] + B_name[5:]
		B1_name = random.sample(self.id2names[B_id], 1)[0]
		A_name_calm = A_name[0:4] + '0' + A_name[5:]
		B_name_calm = B_name[0:4] + '0' + B_name[5:]
		# A_name_calm = A_name[0:4] + '0_4_10' + A_name[10:]
		# B_name_calm = B_name[0:4] + '0_4_10' + B_name[10:]

		A_shape = self.name2shape[A_name]
		B_shape = self.name2shape[B_name]
		B1_shape = self.name2shape[B1_name]
		A_shape_calm = self.name2shape[A_name_calm]
		B_shape_calm = self.name2shape[B_name_calm]

		A_shape = torch.Tensor(A_shape)
		B_shape = torch.Tensor(B_shape)
		B1_shape = torch.Tensor(B1_shape)
		A_shape_calm = torch.Tensor(A_shape_calm)
		B_shape_calm = torch.Tensor(B_shape_calm)

		return [A_shape, B_shape, B1_shape, A_shape_calm, B_shape_calm]

	def __len__(self):

		return len(self.ids)





if __name__ == '__main__':
	# if True:
	# 	trainset = Landmarks_RaFD()
	# 	trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
	# 	print('start ==> loading dataset')
	# 	for batch_idx, (A_shape, B_shape, B1_shape, A_shape_calm, B_shape_calm) in enumerate(trainloader):
	# 		print('show {:>20} ==> {}/{}'.format('batch_idx', batch_idx + 1, len(trainloader)))
	# 		print('show {:>20} ==> {}'.format('A_shape', A_shape.shape))
	# 		print('show {:>20} ==> {}'.format('B_shape', B_shape.shape))
	# 		print('show {:>20} ==> {}'.format('B1_shape', B1_shape.shape))
	# 		print('show {:>20} ==> {}'.format('A_shape_calm', A_shape_calm.shape))
	# 		print('show {:>20} ==> {}'.format('B_shape_calm', B_shape_calm.shape))
	# 		break
	if True:
		trainset = Landmarks_PIE()
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
		print('start ==> loading dataset')
		for batch_idx, (A_shape, B_shape, B1_shape, A_shape_calm, B_shape_calm) in enumerate(trainloader):
			print('show {:>20} ==> {}/{}'.format('batch_idx', batch_idx + 1, len(trainloader)))
			print('show {:>20} ==> {}'.format('A_shape', A_shape.shape))
			print('show {:>20} ==> {}'.format('B_shape', B_shape.shape))
			print('show {:>20} ==> {}'.format('B1_shape', B1_shape.shape))
			print('show {:>20} ==> {}'.format('A_shape_calm', A_shape_calm.shape))
			print('show {:>20} ==> {}'.format('B_shape_calm', B_shape_calm.shape))
			break