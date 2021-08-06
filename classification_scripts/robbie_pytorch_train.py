from TRAIN_NETWORKS import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", help="which data folder to use")
parser.add_argument("--model", help="which model to use [resnet, vgg, alexnet")
parser.add_argument("--batch_size", help="batch size")
parser.add_argument("--num_epochs", help="number of epochs to train for")
parser.add_argument("--learning_rate", help="learning rate to use")
parser.add_argument("--num_classes", help="number of classes in dataset")
parser.add_argument("--test_txt", help="COVIDx test file")
parser.add_argument("--train_txt", help="COVIDx train file")
args = parser.parse_args()

model=args.model
batch_size = int(args.batch_size)
num_epochs = int(args.num_epochs)
learning_rate = float(args.learning_rate)
num_classes=int(args.num_classes)
data_folder = args.data_folder
test_file = args.test_txt
train_file = args.train_txt

save_dir = './out/' + model \
+ '_Jan13_' \
+ '_BS_' +str(batch_size) \
+ '_E_' + str(num_epochs) \
+ '_LR_' + str(learning_rate) \
+ '_TrainF_' + train_file.split('/')[-1] \
+ '_TestF_' + test_file.split('/')[-1] \
+ '_df_' + data_folder.split('/')[-1] 



print(save_dir)
#input()
os.mkdir(save_dir)
#input()
print(os.listdir(data_folder))
#input()
print(open(test_file).readlines())
#input()
print(open(train_file).readlines())
#input()
print(os.listdir(data_folder+'/test'))
#input()
print(os.listdir(data_folder+'/train'))
#input()

test_files_list = set([line.split()[1] for line in open(test_file).readlines()])
train_files_list = set([line.split()[1] for line in open(train_file).readlines()])

test_files_folder = set(os.listdir(os.path.join(data_folder,'test')))
train_files_folder = set(os.listdir(os.path.join(data_folder,'train')))

print('len test files list: ',len(test_files_list))
print('len train files list: ',len(train_files_list))
print('len test files folder: ',len(test_files_folder))
print('len train files folder: ',len(train_files_folder))

print('test files missing in folder:')
print(len(test_files_list.difference(test_files_folder)))
print('train files missing in folder:')
print(len(train_files_list.difference(train_files_folder)))

if len(test_files_list.difference(test_files_folder))!=0:
    print("ERROR")
    exit()
if len(train_files_list.difference(train_files_folder))!=0:
    print("ERROR")
    exit()
    

print('test files missing in list:')
print(len(test_files_folder.difference(test_files_list)))
print('train files missing in list:')
print((train_files_folder.difference(train_files_list)))

print('datset checks out')

Net = Network(model_name = model, num_classes=num_classes, progress_folder=save_dir)


mapping = {'normal':0,'pneumonia':1,'COVID-19':2}
inv_mapping = {0:'normal',1:'pneumonia',2:'COVID-19'}

Net.load_training_data(data_dir=os.path.join(data_folder, 'train'), 
                       text_file=train_file, 
                       mapping = {'normal':0,'pneumonia':1,'COVID-19':2}, 
                       batch_size=batch_size)

Net.load_test_data(data_dir=os.path.join(data_folder, 'test'), 
                   text_file=test_file, 
                   mapping = {'normal':0,'pneumonia':1,'COVID-19':2}, 
                   batch_size=1)

Net.display_batch(1,batch_size,inv_mapping=inv_mapping)
Net.get_optimizer(lr=.0001)
Net.get_loss(weight = [1,1,4])

Net.training_loop(epochs=num_epochs)

Net.log.close()
