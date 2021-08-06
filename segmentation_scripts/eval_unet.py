import torch
print(torch.cuda.is_available())

TEST_PATH_NIH = "../../datasets/segmentation/NIH_test"
TEST_PATH_COHEN = "../../datasets/segmentation/cohen_test"

FOLDER_PATH = "../../segmentation_models/_null_"

from robbie_lib import ROBBIE

network= ROBBIE(D = 5, 
                W = 6, 
                LR = .0001,
                model_name = "_null_",
                data_handle = '_mask',
                save_freq=10, 
                folder = FOLDER_PATH,
                load = "../../segmentation_models/UNET_NIH_A_MODEL/NIH_A_MODEL_BEST.pth")

print(network.device)

print("NIH performance")
network.TEST(PATH=TEST_PATH_NIH)

print("Cohen performance")
network.TEST(PATH=TEST_PATH_COHEN)

import shutil
shutil.rmtree(FOLDER_PATH)

