import torch
import torch.nn as nn
from model import Generate_quarter,Generate_quarter_refine,Generate,Lap
from torchvision.models import vgg16
from perceptual import LossNetwork
from dataloader import Dataset
from torch.utils.data import DataLoader
import time
from utils import adjust_learning_rate,to_psnr
import torch.nn.functional as F
"""init arguments"""
lr=1e-4
batch_size=12
network_height=3
network_width=6
num_dense=4
growth_rate=16 #RDB
lambda_loss=0.04
root_dir="./dataset/"
device=torch.device("cuda:0")
train_phrase=1
EPOCH=500
if __name__ == '__main__':
    if train_phrase==1:
        net=Generate_quarter(height=network_height,width=network_width,num_dense_layer=num_dense,growth_rate=growth_rate)
        optimizer=torch.optim.Adam(list(net.parameters()),lr=lr,betas=(0.5,0.999))
        net=net.to(device)
    if train_phrase==2:
        net=Generate_quarter(height=network_height,width=network_width,num_dense_layer=num_dense,growth_rate=growth_rate)
        G2=Generate_quarter_refine(height=network_height,width=network_width,num_dense_layer=num_dense,growth_rate=growth_rate)
        optimizer = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.999))
        net=net.to(device)
        G2=G2.to(device)
        net.load_state_dict(torch.load('./checkpoint/1_615.tar'))
        G2.load_state_dict(torch.load('./checkpoint/1_615.tar'))
        params=list(net.parameters())+list(G2.parameters())
    if train_phrase==3:
        net=Generate_quarter(height=network_height,width=network_width,num_dense_layer=num_dense,growth_rate=growth_rate)
        G2=Generate_quarter_refine(height=network_height,width=network_width,num_dense_layer=num_dense,growth_rate=growth_rate)
        G3=Generate(height=network_height,width=network_width,num_dense_layer=num_dense,growth_rate=growth_rate)
        net=net.to(device)
        G2=G2.to(device)
        G3=G3.to(device)
        net.load_state_dict('./checkpoint/2_1810_G1.tar')
        G2.load_state_dict('./checkpoint/2_1810_G2.tar')
        G3.load_state_dict('./checkpoint/33_35_G3.tar')
        params=list(net.parameters())+list(G2.parameters())+list(G3.parameters())
        optimizer=torch.optim.Adam(params,lr=lr,betas=(0.5,0.999))

    vgg_model=vgg16(pretrained=True).features[:16]
    vgg_model=vgg_model.to(device)
    for param in vgg_model.parameters():
        param.requires_grad=False
    loss_network=LossNetwork(vgg_model)
    loss_network.eval()
    loss_lap=Lap()
    start_epoch=0
    loss_rec1=nn.SmoothL1Loss()
    loss_rec2=nn.MSELoss()
    num=0
    avg=nn.AvgPool2d(3,stride=2,padding=1)
    train_data=Dataset("./dataset/RICE1/")
    train_dataloader=DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True)
    for epoch in range(start_epoch,EPOCH):
        psnr_list=[]
        start_time=time.time()
        adjust_learning_rate(optimizer,epoch)

        for batch_id,train_data in enumerate(train_dataloader):
            cloud,gt=train_data
            optimizer.zero_grad()
            cloud=cloud.to(device)
            gt=gt.to(device)
            gt_quarter_1=F.interpolate(gt,scale_factor=0.25,recompute_scale_factor=True)
            gt_quarter_2=F.interpolate(gt,scale_factor=0.25,recompute_scale_factor=True)

            if train_phrase==1:
                decloud_1,feat_extra_1=net(cloud)
                rec_loss1=loss_rec1(decloud_1,gt)
                perceptual_loss=loss_network(decloud_1,gt)
                lap_loss=loss_lap(decloud_1,gt)
                psnr=to_psnr(decloud_1,gt)
                psnr_list.extend(psnr)

            if train_phrase==2:
                decloud_1,feat_extra_1=net(cloud)
                decloud_2,feat_extra_2=G2(decloud_1)
                rec_loss1=(loss_rec1(decloud_2,gt)+loss_rec1(decloud_1,gt))/2.0
                rec_loss2=loss_rec2(decloud_2,gt)
                perceptual_loss=loss_network(decloud_2,gt)
                lap_loss=loss_lap(decloud_2,gt)
                psnr=to_psnr(decloud_2,gt)
                psnr_list.extend(psnr)

            if train_phrase==3:
                decloud_1,feat_extra_1=net(F.interpolate(cloud,scale_factor=0.25,recompute_scale_factor=True))
                decloud_2,feat,feat_extra_2=G2(decloud_1)
                decloud=G3(cloud,F.interpolate(decloud_2,scale_factor=4,recomput_scale_factor=True),feat)
                rec_loss1=(loss_rec1(decloud,gt)+loss_rec1(decloud_2,gt_quarter_2)+loss_rec1(decloud_1,gt_quarter_1))/3.0
                rec_loss2=loss_rec2(decloud,gt)
                perceptual_loss=(loss_network(decloud,gt)+loss_network(F.interpolate(decloud,scale_factor=0.5,recompute_scale_factor=True),F.interpolate(gt,scale_factor=0.5,recomput_scale_factor=True))+loss_network(F.interpolate(decloud,scale_factor=0.25,recompute_scale_factor=True),F.interpolate(gt,scale_factor=0.25,recompute_scale_factor=True))+loss_network(decloud_2,gt_quarter_2))/4.0
                lap_loss=loss_lap(decloud,gt)
                psnr=to_psnr(decloud,gt)
                psnr_list.extend(psnr)

            loss=rec_loss1*1.2+0.04*perceptual_loss
            loss.backward()
            optimizer.step()

            print(f'epoch={epoch} | loss={loss:.4f} | PSNR={psnr:.4f}')

        psnr_avg=sum(psnr_list)/len(psnr_list)
        print(f'EPOCH{epoch} train finish, Aver PSNR is {psnr_avg:.4f}')

        if epoch%5==0:
            if train_phrase==1:
                torch.save(net.state_dict(),'./checkpoint/'+str(int(train_phrase))+'_'+str(epoch)+'.tar')
            if train_phrase==2:
                torch.save(net.state_dict(), './checkpoint/' + str(int(train_phrase)) + '_' + str(epoch) + '_G1.tar')
                torch.save(G2.state_dict(), './checkpoint/' + str(int(train_phrase)) + '_' + str(epoch) + '_G2.tar')
            if train_phrase==3:
                torch.save(net.state_dict(), './checkpoint/' + str(int(train_phrase)) + '_' + str(epoch) + '_G1.tar')
                torch.save(G2.state_dict(), './checkpoint/' + str(int(train_phrase)) + '_' + str(epoch) + '_G2.tar')
                torch.save(G3.state_dict(), './checkpoint/' + str(int(train_phrase)) + '_' + str(epoch) + '_G3.tar')





