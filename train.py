import torch
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
from torch.autograd import Variable
import PoseNet
from DataSet import *

learning_rate = 0.0001
batch_size = 75
EPOCH = 80000
directory = '/path_to_dataset/KingsCollege/'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    # train dataset and train loader
    datasource = DataSource(directory, train=True)
    train_loader = Data.DataLoader(dataset=datasource, batch_size=batch_size, shuffle=True)

    # posenet
    posenet = PoseNet.posenet_v1().to(device)
    # posenet.cuda()

    # loss function
    criterion = PoseNet.PoseLoss(0.3, 150, 0.3, 150, 1, 500)

    # load pre-trained model

    # train the network
    optimizer = torch.optim.SGD(nn.ParameterList(posenet.parameters()), lr=learning_rate)

    for epoch in range(EPOCH):
        for step, (images, poses) in enumerate(train_loader):
            b_images = Variable(images, requires_grad=True).to(device)
            poses[0] = np.array(poses[0])
            poses[1] = np.array(poses[1])
            poses[2] = np.array(poses[2])
            poses[3] = np.array(poses[3])
            poses[4] = np.array(poses[4])
            poses[5] = np.array(poses[4])
            poses[6] = np.array(poses[5])
            poses = np.transpose(poses)
            b_poses = Variable(torch.Tensor(poses), requires_grad=True).to(device)

            p1_x, p1_q, p2_x, p2_q, p3_x, p3_q = posenet(b_images)
            loss = criterion(p1_x, p1_q, p2_x, p2_q, p3_x, p3_q, b_poses)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 20 == 0:
                print("iteration: " + str(epoch) + "\n    " + "Loss is: " + str(loss))


if __name__ == '__main__':
    main()
