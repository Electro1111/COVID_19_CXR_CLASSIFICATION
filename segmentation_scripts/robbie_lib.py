import sys
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics as metrics
from unet import UNet
import os
import random
import time
import shutil
import matplotlib.pyplot as plt
import random
class ROBBIE:
	def __init__(self,D,W, reset_optim = False, model_name = 'UNET', LR = .0001, load = None, data_handle = '_LABEL',save_freq = 10, folder = 'folder'):
		
		self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
		self.D = D
		self.W = W
		self.load = load
		self.LR = LR
		self.data_handle = data_handle
		self.save_freq = save_freq
		self.e = 0
		self.i = 0
		self.model_name = model_name  
		self.folder = folder
		self.init_time = time.time()
		self.reset_optim = reset_optim
		self.Model_Init()
		self.init_progress_track()
		      

		self.meanf1s=[]
		self.max_f1=-1
		self.loss = 0

        
	def Model_Init(self):
		print('initializing model')
		os.mkdir(self.folder) 
		os.mkdir(os.path.join(self.folder,'epochs'))
		if self.load == None:
			print('from blank slate')
			self.model = 0
			self.model = UNet(n_classes=2, padding=True, up_mode='upconv', depth=self.D,wf=self.W).to(self.device)
			self.optim = torch.optim.Adam(self.model.parameters(), self.LR)
            
		else:
			print('from saved model')
			self.model = UNet(n_classes=2, padding=True, up_mode='upconv', depth=self.D,wf=self.W).to(self.device)
			self.optim = torch.optim.Adam(self.model.parameters(), self.LR)			
			self.load_model()
            
		self.criterion = nn.CrossEntropyLoss()


	def GetDataSet(self, data_path):
		print('loading data from: ', data_path)        
		files = os.listdir(data_path)
		#print(files)   
		All_Data=[]	
		for file in files:
			if self.data_handle in file:
				dname = file.split(self.data_handle)[0]+file.split(self.data_handle)[1]
				lname = file
				dpath = os.path.join(data_path,dname)
				lpath = os.path.join(data_path,lname)
				d = cv2.imread(dpath,0)
				l = cv2.imread(lpath,0)
				#print('label file:\n', lpath)
				#print('img data file:\n', dpath)
				l = (l>0).astype(int)
				All_Data.append((d,l,file))
		random.shuffle(All_Data)
		return All_Data
    

	def init_progress_track(self):
		cols = 'time','epoch','F1_0','F1_1'
		self.metrics_df = pd.DataFrame(columns=cols)

	
	def update_progress_track(self):
		f1_0,f1_1=self.Validate()
        
		if f1_1>self.max_f1:
			print("New Best Score Record!!!")
			self.max_f1=f1_1
			self.save_best_model()
            
		data_to_append = {'F1_0':f1_0, 'F1_1':f1_1, 'epoch':self.e, 'time':time.time()-self.init_time}
		print('\nDICE C0 = ',round(f1_0,3),'\tDICE C1 = ',round(f1_1,3))
		self.metrics_df = self.metrics_df.append(data_to_append, ignore_index = True)

	def save_progress_track(self):
		self.metrics_df.to_csv(os.path.join(self.folder,self.model_name+'.csv'))

	def PreProcess(self,x):
		x = x-np.mean(x)
		x = x/np.std(x)
		return x
		
	def train(self,epochs=300):
		for e in range(epochs):
			self.e = e
			if (self.e!=0)&(self.e%self.save_freq==0):
				self.save_model()
			self.update_progress_track()
			self.save_progress_track()
			random.shuffle(self.training_data)
			for i in range(len(self.training_data)):
				self.i=i
				self.iterate()
				self.text = "\r"+'model:'+str(self.model_name)+'\t\tepoch:'+str(self.e)+'\t\titeration:'+str(self.i)+'/'+str(len(self.training_data))+'\t\tloss:'+str(round(self.loss.item(),4))

				sys.stdout.write(self.text)
				sys.stdout.flush()
	def iterate(self):
		x,gt,f = self.training_data[self.i]
		x = self.PreProcess(x)
		x = np.reshape(x,[1,1,x.shape[0],x.shape[1]])
		gt = np.reshape(gt,[1,gt.shape[0],gt.shape[1]])
		x = torch.from_numpy(x).float()
		gt=torch.from_numpy(gt).long()
		x = x.to(self.device)
		gt = gt.to(self.device)
		self.optim.zero_grad()
		y = self.model(x)
		loss = self.criterion(y, gt)
		loss.backward()
		self.optim.step()
		self.loss = loss



	def Validate(self):
		print('evaluating')
		l,w = (self.test_data[0][0]).shape


		y_pred = np.zeros([l,w,self.test_sample_size])
		y_true = np.zeros([l,w,self.test_sample_size]) 
		images = []
		index = 0
        
		t0 = time.time() 

		for x,gt,name in random.sample(self.test_data,self.test_sample_size):
			print(name)
			prediction = self.infer(x)
			y_pred[:,:,index]=prediction
			y_true[:,:,index]=gt
			index+=1
			images.append((x,prediction,gt))


		check = self.stich(images[:4])
		e_folder = os.path.join(self.folder,'epochs')
		cv2.imwrite(os.path.join(e_folder,str(self.e)+'.png'),check)
		print('computing dice')
		y_pred = np.ravel(y_pred)
		y_true = np.ravel(y_true)
		t0 = time.time()
		f1_0,f1_1 = metrics.f1_score(y_true=y_true,y_pred=y_pred,labels = [0,1],average=None)

		return f1_0,f1_1

	def infer(self,x):
		x = self.PreProcess(x)
		x = np.reshape(x, [1,1,x.shape[0],x.shape[1]])
		x = torch.from_numpy(x).float()
		x = torch.cat((x, x), 0).to(self.device)
		y_out = self.model(x)
		y_out = y_out[0].squeeze().cpu()
		prediction = torch.argmax(y_out,dim=0).numpy()
		return prediction
	def to_int8(self,img):
		img = img-np.min(img)
		img = img/np.max(img)
		img = img*255
		img = img.astype(int)
		return img

	def stich(self, imgs):
		x,p,gt = imgs[0]
		x=self.to_int8(x)
		gt=self.to_int8(gt)
		p=self.to_int8(p)
		stitch=np.concatenate((x,p,gt),axis = 1)	
		for x,p,gt in imgs[1:]:
			x=self.to_int8(x)
			gt=self.to_int8(gt)
			p=self.to_int8(p)
			add =np.concatenate((x,p,gt),axis = 1)
			stitch = np.concatenate((stitch,add),axis = 0)
		return stitch
	
	def save_model(self):
		print('\nsaving model')
		torch.save({'epoch': self.e,'model_state_dict': self.model.state_dict(),'optimizer_state_dict': self.optim.state_dict(),'loss': self.loss}, os.path.join(self.folder,self.model_name+'.pth'))

	def save_best_model(self):
		print('\nsaving BEST model')
		torch.save({'epoch': self.e,'model_state_dict': self.model.state_dict(),'optimizer_state_dict': self.optim.state_dict(),'loss': self.loss}, os.path.join(self.folder,self.model_name+'_BEST.pth'))

	def load_model(self):
		print('loading model from', self.load)
		checkpoint = torch.load(self.load)
		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.e = 0
		self.l = 0
		if self.reset_optim==False:
			self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
			self.e = checkpoint['epoch']
			self.l = checkpoint['loss']
			print('loading optimizer')
            
            
	def TEST(self, PATH, show=False):
		data = self.GetDataSet(PATH)
        
        
		c1_scores_dict = {"recall":[],"precision":[],"jaccard":[],"f1":[],"accuracy":[]}
        
		c0_scores_dict = {"recall":[],"precision":[],"jaccard":[],"f1":[],"accuracy":[]}
        
        
            
		for x,gt,f in data:
            
            
            
			result = self.infer(x)
			y_pred = np.ravel(result)
			y_true = np.ravel(gt)
            

			if show==True:
                
				plt.figure()
				plt.suptitle(f)
				plt.subplot(1,3,1)   
				plt.imshow(x)
				plt.subplot(1,3,2)
				plt.imshow(gt)
				plt.subplot(1,3,3)
				plt.imshow(result)
				plt.show()



			for _metricname_,_score_ in [("recall", metrics.recall_score),("precision", metrics.precision_score),("jaccard", metrics.jaccard_score),("f1",metrics.f1_score),("accuracy",metrics.accuracy_score)]:
                
                
            
				if _metricname_=="accuracy":
					class_0=class_1 = _score_(y_true=y_true,y_pred=y_pred)
                
				else:
					class_0,class_1 = _score_(y_true=y_true,y_pred=y_pred,labels = [0,1],average=None)

                
				c0_scores_dict[_metricname_].append(class_0)
				c1_scores_dict[_metricname_].append(class_1)
                
				if show==True:
					print(_metricname_,class_0,class_1)



                
        

		print("CLASS 0 metrics:")
        
		for metric,scores in c0_scores_dict.items():
			print(metric, ": ", np.mean(scores))
        
		print("CLASS 1 metrics:")
        
		for metric,scores in c1_scores_dict.items():
			print(metric, ": ", np.mean(scores))
        
        
        
        
        
        
        

        

        

	def TrainTest(self, TRAIN_PATH, TEST_PATH, epochs, test_sample_size):
		self.test_sample_size=test_sample_size
		self.training_data = self.GetDataSet(TRAIN_PATH)
		self.test_data=self.GetDataSet(TEST_PATH)
		self.model_name=self.model_name
		self.train(epochs)
        
        

