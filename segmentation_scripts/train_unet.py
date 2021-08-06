import torch
print(torch.cuda.is_available())

TRAIN_PATH = "../../datasets/segmentation/train"

TEST_PATH = "../../datasets/segmentation/NIH_test"
FOLDER_PATH = "../../segmentation_models/UNET_NIH_A_MODEL"

from robbie_lib import ROBBIE
network= ROBBIE(D = 5, W = 6, LR = .0001,model_name = "NIH_A_MODEL",data_handle = '_mask',save_freq=10, folder = FOLDER_PATH)
print(network.device)
network.TrainTest(TRAIN_PATH, TEST_PATH, 50, test_sample_size = 100)