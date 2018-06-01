# Pytorch-PoseNet

**This Repository is a [PoseNet](http://mi.eng.cam.ac.uk/projects/relocalisation/) implementation for pytorch.**

As described in the ICCV 2015 paper **PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization** Alex Kendall, Matthew Grimes and Roberto Cipolla [http://mi.eng.cam.ac.uk/projects/relocalisation/]

## Usage

 - Download the Cambridge Landmarks King's College dataset [from here.](https://www.repository.cam.ac.uk/handle/1810/251342)

 - The PoseNet model is defined in the PoseNet.py file

 - The starting and trained weights (posenet.npy and PoseNet.ckpt respectively) for training were obtained by converting tensorflow-model weights [from here](https://drive.google.com/file/d/0B5DVPd_zGgc8ZmJ0VmNiTXBGUkU/view?usp=sharing) and then training.

 - To run:
   - Extract the King's College dataset to wherever you prefer
   - Extract the starting and trained weights to wherever you prefer
   - Update the paths on line 12 (train.py)
   - If you want to retrain, simply run train.py (note this will take a long time)
