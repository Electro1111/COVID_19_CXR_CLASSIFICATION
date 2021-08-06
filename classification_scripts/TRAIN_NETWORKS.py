from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import sys
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
import sklearn.metrics as skm
import pandas as pd
from robbie_pytorch_dataset import *

class Network:
    def __init__(self, model_name = "resnet", num_classes = 3, progress_folder = "torchvision_model_training"):
        self.get_device()
        self.num_classes = num_classes
        self.progress_folder = progress_folder
        self.log = open(os.path.join(self.progress_folder, 'training_log.txt'), 'w')
        self.init_model(model_name=model_name,num_classes = num_classes)
        self.epoch = 0
        self.loss = 0
        self.best_acc = 0
        self.metrics_hist = []
        self.save_path = os.path.join(progress_folder,'model.pth')
        self.history_data_frame = pd.DataFrame(columns = ['accuracy','balanced_accuracy', 'f1', 'precision', 'recall','time'])
        
        self.train_loss_hist=[]
        self.train_acc_hist=[]
        
        self.test_loss_hist=[]
        self.test_acc_hist=[]
        
    
    
    def init_model(self, model_name, num_classes, use_pretrained=True):
        print("initializing model")
        self.log.write("\n"+"initializing model")
        model_ft = None
        input_size = 0
        
        

        if model_name == "resnet":
            """ Resnet50
            """
            print("loading resnet with n class = ", num_classes)
            self.log.write("\n"+"loading resnet with n class = "+ str(num_classes))
            model_ft = models.resnet50(pretrained=use_pretrained)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            self.input_size = 224

        elif model_name == "alexnet":
            """ Alexnet
            """
            print("loading alexnet with n class = ", num_classes)
            self.log.write("\n"+"loading alexnet with n class = "+ str(num_classes))
            model_ft = models.alexnet(pretrained=use_pretrained)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
            self.input_size = 224

        elif model_name == "vgg":
            """ VGG11_bn
            """
            print("loading vgg with n class = ", num_classes)
            self.log.write("\n"+"loading vgg with n class = "+ str(num_classes))
            model_ft = models.vgg11_bn(pretrained=use_pretrained)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
            self.input_size = 224
            
        else:
            print("not a valid model, please choose between alexnet, vgg, and resnet")
            self.log.write("\n"+"invalid model, please choose correct name")
        self.model = model_ft.to(self.device)

        
    
    def load_training_data(self, data_dir, text_file, mapping, batch_size):
        
        print("loading training data from: ", data_dir)
        self.log.write("\n"+"loading train data from: "+data_dir)
        
        trnsfms = [
        robbie_croptop_transform(.08),
        robbie_central_crop(),
        robbie_resize(224),
        robbie_rand_ratio_resize(prob = .3, delta = .1),
        transforms.ColorJitter(brightness=.1, contrast=0, saturation=0, hue=0),
        transforms.RandomAffine(degrees=(10), translate=(.1,.1), scale=None, shear=None, resample=0, fillcolor=0),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.15), ratio=(1,1), interpolation=2),
        transforms.ToTensor()
        ]
        wts = GetSamplerWeights(text_file = text_file)
        
        covid_dataset = COVID_DATASET(text_file = text_file, 
                            data_dir = data_dir,
                            mapping = mapping,
                            transform = transforms.Compose(trnsfms)
                            )
        
        
        
        
        balanced_sampler = torch.utils.data.sampler.WeightedRandomSampler(wts, len(wts)) 
        
        self.train_dataloader = DataLoader(covid_dataset, 
                        batch_size=batch_size,
                        num_workers=0,
                       sampler = balanced_sampler)
        print("done")
        
        self.log.write("\n"+"done")
        

        
        
    def load_test_data(self, data_dir, text_file, mapping, batch_size):
        trnsfms = [
        robbie_croptop_transform(.08),
        robbie_central_crop(),
        robbie_resize(224),
        transforms.ToTensor()
        ]
        
        covid_dataset = COVID_DATASET(text_file = text_file, 
                            data_dir = data_dir,
                            mapping = mapping,
                            transform = transforms.Compose(trnsfms)
                            )
        
        
        self.test_dataloader = DataLoader(covid_dataset, 
                        batch_size=1,
                        num_workers=0,
                        shuffle = False)
        
        
        
        print("loading test data from: ", data_dir)
        self.log.write("\n"+"loading test data from: "+data_dir)
        
        
        
    def get_optimizer(self, lr = .0001):
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    
    def get_loss(self, weight = [1,1,1]):
        
        print('class weights:',weight)
        weight = torch.FloatTensor(weight).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight = weight)
        
    def get_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        
        
    def train_step(self):
        print("train step...")
        self.log.write("\n"+"train step...")
        self.model.train()
        total_loss = 0.0
        total_corrects = 0
        times=[]
        
        for iteration, batch in enumerate(self.train_dataloader):
            t = time.time()
            inputs = batch['image']
            labels = batch['label']
            file_name = batch['file_name']
            print(file_name)
            
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                _, preds = torch.max(outputs, 1)
                self.loss=loss    
            total_loss += loss.item() * inputs.size(0)
            total_corrects +=torch.sum(preds==labels.data)
            
            times.append(t)
            #print(times)
            #print(len(times))
            if len(times)>1:
                avg_iter_time = np.mean([times[i+1]-times[i] for i in range(len(times)-1) ])
                print('estimated epoch time: ', len(self.train_dataloader)*avg_iter_time)
                print('train step epoch eta: ', (len(self.train_dataloader)-iteration)*avg_iter_time)
                print('iter: ', iteration ,'/', len(self.train_dataloader))
                print('iter time:', times[-1]-times[-2])
        print()
        
        nsamples = len(self.train_dataloader.dataset)
        epoch_loss=total_loss/nsamples
        epoch_acc=total_corrects.double()/nsamples
        
        print('train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        self.log.write("\n"+'train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        
        
        self.train_loss_hist.append(epoch_loss)
        self.train_acc_hist.append(epoch_acc)
                
           

    def evaluate_model(self, save = True):
        print("evaluating model....")
        self.log.write("\n"+"evaluating model....")
        
        self.model.eval()
        
        y_pred = []
        y_true =[]
        total_loss = 0.0
        total_corrects = 0
        for iteration, batch in enumerate(self.test_dataloader):
            
            inputs = batch['image']
            labels = batch['label']
            file_name = batch['file_name']
            print(file_name)
            
            
            #sys.stdout.write("\r iter:"+str(iteration)+" of \t"+ str(len(self.test_dataloader)))
            
            #sys.stdout.flush()
            
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
    
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
            total_loss += loss.item() * inputs.size(0)
            total_corrects +=torch.sum(preds==labels.data)
            
            y_pred.append(preds.cpu().numpy())
            y_true.append(labels.data.cpu().numpy())
            
            
        print()
        nsamples = len(self.test_dataloader.dataset)
        epoch_loss=total_loss/nsamples
        epoch_acc=total_corrects.double()/nsamples
        
        
        print('test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        self.log.write("\n"+'test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        
        
        y_pred_temp =[]
        y_true_temp = []

        for row in y_pred:
            for item in row:
                y_pred_temp.append(item)
        for row in y_true:
            for item in row:
                y_true_temp.append(item)

        y_pred=y_pred_temp
        y_true = y_true_temp
        labels=[i for i in range(self.num_classes)]

        accuracy = skm.accuracy_score(y_pred=y_pred, y_true = y_true)
        balanced_accuracy = skm.balanced_accuracy_score(y_pred=y_pred, y_true = y_true)
        precisions=skm.precision_score(y_pred=y_pred, y_true = y_true, labels=labels, average =None)
        recalls=skm.recall_score(y_pred=y_pred, y_true = y_true, labels=labels, average =None)
        f1s = skm.f1_score(y_pred=y_pred, y_true = y_true, labels=labels, average =None)

        '''print('accuracies: '+str(accuracy))
        print('balanced accuracy: '+str(balanced_accuracy))
        print('precisions: '+str(precisions))
        print('recalls: '+ str(recalls))
        print('f1: '+ str(f1s))'''
        
        addition = {'accuracy':accuracy,
                            'balanced_accuracy':balanced_accuracy, 
                            'f1':[f1s.tolist()], 
                            'precision':[precisions.tolist()], 
                            'recall':[recalls.tolist()],
                           'time':time.time()
                   }
        
        new_d = pd.DataFrame(addition)
        self.history_data_frame=self.history_data_frame.append(new_d, ignore_index=True)
        
        
        
        if (epoch_acc>self.best_acc)&save==True:
            print("new best accuracy for model")
            self.log.write("\n"+"new best! saving")
            print("saving model")
            self.best_acc=epoch_acc
            self.best_model = copy.deepcopy(self.model.state_dict())
            self.save_model()
        self.test_loss_hist.append(epoch_loss)
        self.test_acc_hist.append(epoch_acc)    
        
        
    def training_loop(self, epochs=10):
        print("running through training data")
        self.log.write("\n"+"running through training data")
        print("initial evaluation")
        print()
        self.evaluate_model()
        print()
        
        for epoch in range(epochs):
            t0=time.time()
            print("epoch: ", epoch)
            self.log.write("\n\n"+"epoch: "+str(epoch))
            self.epoch = epoch
            self.train_step()
            self.evaluate_model()
            print()
            t1=time.time()
            print('epoch time = ', t1-t0, 'seconds')
            self.history_data_frame.to_csv(os.path.join(self.progress_folder,'training_history.csv'))
            
        
    def save_model(self):
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
            }, self.save_path)
        
    def load_model(self, PATH):
        print("loading model from: ", PATH)
        checkpoint = torch.load(PATH)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']

        
    def revert_to_best_model(self):
        self.model.load_state_dict(self.best_model)
        

    
        
    def display_batch(self, n,  batch_size, inv_mapping):
        
        
        for j, batch in enumerate(self.train_dataloader):
            image_batch, y_batch, file_name = batch['image'],batch['label'],batch['file_name']
            
            batch_size = len(image_batch)
           
            plt.figure(figsize = (10,10))
            print(file_name)
            for i in range(image_batch.shape[0]):
                img = image_batch[i].permute(1, 2, 0).numpy()
                fname = str(file_name[i])


                plt.subplot(int(batch_size/8)+1,8,i+1)
                plt.imshow(img[:,:,0])
                #print(np.max(img))
                plt.title(str(inv_mapping[int(y_batch[i])]))
                plt.text(0, -0.1, fname)
            plt.savefig(os.path.join(self.progress_folder,'train_batch.png'))
            if j==n-1:
                break
                
            
        for j, batch in enumerate(self.test_dataloader):
            image_batch, y_batch, file_name = batch['image'],batch['label'],batch['file_name']
            
            batch_size = len(image_batch)
           
            plt.figure(figsize = (10,10))
            plt.suptitle("test dataloader batch = "+str(j))
            print(file_name)
            for i in range(image_batch.shape[0]):
                img = image_batch[i].permute(1, 2, 0).numpy()
                fname = str(file_name[i])


                plt.subplot(1,int(batch_size/8)+1,i+1)
                plt.imshow(img[:,:,0])
                #print(np.max(img))
                plt.title(str(inv_mapping[int(y_batch[i])]))
                plt.text(0, -0.1, fname)
            plt.savefig(os.path.join(self.progress_folder,'test_batch.png'))
            if j==n*batch_size:
                break
            
        
    
        
        
    
        
        