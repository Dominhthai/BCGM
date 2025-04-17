
import argparse
import torch
import numpy as np
import random
import json
import os
from os.path import join
import sys
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.Classifier import Classifier
from config import Config
# from dataset.KS import KSDataset
from dataset.CramedDataset import CramedDataset
# from utils.log_file import Logger
from datetime import datetime

from dataset.spatial_transforms import get_spatial_transform,get_val_spatial_transforms
from dataset.loader import VideoLoaderHDF5

from sklearn.metrics import average_precision_score
import torch.nn.functional as F

TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now()) 


def get_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', default='CREMAD', type=str,
                        help='VGGSound, KineticSound, CREMAD, AVE')
    parser.add_argument('--fps', default=1, type=int)
#     parser.add_argument('--audio_path', default='/kaggle/working/BML_TPAMI2024/AudioWAV', type=str)
#     parser.add_argument('--visual_path', default='/kaggle/working/BML_TPAMI2024', type=str)
    parser.add_argument('--audio_path', default='/kaggle/input/cremad/AudioWAV/AudioWAV', type=str)
    parser.add_argument('--visual_path', default='/kaggle/input/cremad/Image-01-FPS', type=str)
    parser.add_argument('--use_modulation',action='store_true',help='use gradient modulation')
    parser.add_argument('--use_adam_drop',action='store_true',help='use adam-drop')
    parser.add_argument('--modulation', default='OGM_GE', type=str,choices=['Normal', 'OGM', 'OGM_GE'])
    parser.add_argument('--use_OGM_plus',action='store_true')
    parser.add_argument('--fusion_method', default='concat', type=str,choices=['sum', 'concat', 'gated'])
    parser.add_argument('--train', action='store_true', help='turn on train mode')
    parser.add_argument('--resume_model',action='store_true',help='whether to resume model')
    parser.add_argument('--resume_model_path')
    parser.add_argument('--q_base',type=float,default=0.5)
    parser.add_argument('--lam',type=float,default=0.5)
    parser.add_argument('--p_exe',type=float,default=0.7)
    parser.add_argument('--alpha',type=float,default=1.0)
    parser.add_argument('--modulation_starts',type=int,default=0)
    parser.add_argument('--modulation_ends',type=int,default=80)
    parser.add_argument('--audio_drop',type=float,default=0.0)
    parser.add_argument('--visual_drop',type=float,default=0.0)
    parser.add_argument('--exp_name',type=str,default='exp')

    # Dropouts
    parser.add_argument('--attn_dropout', type=float, default=0.15, help='attention dropout')
    parser.add_argument('--relu_dropout', type=float, default=0.15,
                        help='relu dropout')
    parser.add_argument('--embed_dropout', type=float, default=0.2,
                        help='embedding dropout')
    parser.add_argument('--res_dropout', type=float, default=0.1,
                        help='residual block dropout')
    parser.add_argument('--out_dropout', type=float, default=0.1,
                        help='output layer dropout')
    
    # Architecture
    parser.add_argument('--nlevels', type=int, default=5,
                        help='number of layers in the network')
    parser.add_argument('--cls_layers', type=int, default=2,
                        help='number of layers in the network')
    parser.add_argument('--num_heads', type=int, default=5,
                        help='number of heads for the transformer network')
    parser.add_argument('--proj_dim', type=int, default=40,
                        help='number of heads for the transformer network')
    parser.add_argument('--attn_mask', action='store_false',
                        help='use attention mask for Transformer')


    # Tuning
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='batch size')
    parser.add_argument('--clip', type=float, default=0.8,
                        help='gradient clip value')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--cls_lr', type=float, default=5e-4,
                        help='classifier learning rate')
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of epochs')
    parser.add_argument('--when', type=int, default=10,
                        help='when to decay learning rate')
    parser.add_argument('--rou', type=float, default=1.3)
    parser.add_argument('--lamda', type=float, default=0.2)


    # Logistics
    parser.add_argument('--log_interval', type=int, default=30,
                        help='frequency of result logging')
    parser.add_argument('--seed', type=int, default=666,
                        help='random seed')
    parser.add_argument('--no_cuda', action='store_true',
                        help='do not use cuda')

    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

weight_a=0.36
weight_v=0.27
weight_av=0.37

def train(cfg,epoch,model,device,dataloader,optimizer,scheduler,tb=None):
    # cls_grad and fusion_grad are weights, not gradients :)
    def cal_cos(cls_grad, fusion_grad): # always remember cls_grad*2=fusion_grad
        fgn = fusion_grad.clone().view(-1) # Convert into 1D array- [B,C]=[6,512]=>[3072]
        cost = list()
        for i in range(len(cls_grad)):
            tmp = cls_grad[i].clone().view(-1) # convert into 1D array -[B,C]=[6,1024]=>[6144]
            
            # padding shorter vector with zeros, which is tmp
            pad_length=len(fgn)-len(tmp)
            tmp=torch.cat([tmp,torch.zeros(pad_length, device=tmp.device)])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
            
            l = F.cosine_similarity(tmp, fgn, dim=0) 
            cost.append(l)
        return cost


    loss_fn=nn.CrossEntropyLoss().to(device)
    relu=nn.ReLU(inplace=True)
    tanh=nn.Tanh()
    model.train()
    total_loss=0
    total_loss_1=0
    total_loss_2=0
    with tqdm(total=len(dataloader), desc=f"Train-epoch-{epoch}") as pbar:
        for step, (spec,image,label) in  enumerate(dataloader):
            spec=spec.to(device) # b,h,w (64, 257, 300)
            image=image.to(device) # b,c,t,h,w
            label=label.to(device)
            optimizer.zero_grad()
            warm_up=1 if epoch<=5 else 0
            
            # warm_up=0
            out_1,out_2,out,update_flag,performance_1,performance_2, cls_grad=model(spec.unsqueeze(1).float(),image.float(),label,warm_up) # spec: (64, 1, 257, 300)-squeeze 1st 

            if warm_up==0 and cfg.use_adam_drop:
                if torch.sum(update_flag,dim=0)==0:
                    continue
                select_mask=update_flag!=0
                label=label[select_mask]
                out_1=out_1[select_mask]
                out_2=out_2[select_mask]
            

            loss=loss_fn(out,label)
            loss_1=loss_fn(out_1,label)
            loss_2=loss_fn(out_2,label)
            total_loss+=loss.item()
            total_loss_1+=loss_1.item()
            total_loss_2+=loss_2.item()

            # if warm_up==0:
            #     loss=loss*weight_av+loss_1*weight_a+loss_2*weight_v
            
            if cfg.l_gm is not None:
                loss += cfg.lamda * cfg.l_gm
            loss.backward()

            # Get the gradient of fusion module (actually weights)
            for name, para in model.named_parameters():
                if 'fxy.weight' in name:
                    # fusion_grad = para.grad # with gradient
                    fusion_grad = para # with weights

            llist = cal_cos(cls_grad, fusion_grad) # Calculate sim(classifiers_grad, fusion_grad) using cosine similarity
            # print(f"llist[0] Shape: {llist[0].shape}, llist[0]: {llist[0]}")
            
            diff = [performance_1-cfg.pre_perform[0], performance_2-cfg.pre_perform[1]] # text, audio
            # performance_1 and performance_2 are tensors
            # print ("performance_1: ", performance_1)
            # print("performance_2: ", performance_2)
            # print("diff: ", diff)
            diff_sum = sum(diff) + 1e-3
            
            coeff = list()
            # Calculate gradient magnitude Bt
            for d in diff:
                coeff.append((diff_sum - d) / diff_sum)
            coeff = np.array([t.cpu().detach().numpy() for t in coeff])
            # print("coeff: ", coeff)

            # Update performance previous step and l_gm
            cfg.pre_perform[0] = performance_1
            cfg.pre_perform[1] = performance_2

            cfg.l_gm = np.sum(np.abs(coeff)) - (coeff[0] * llist[0]+ coeff[1] * llist[1])
            cfg.l_gm /= cfg.num_mod

            # Update parameters of each encoder's modality θϕi
            for i in range(cfg.num_mod):
                for name, params in model.named_parameters():
                    if f'encoders.{i}' in name:
                        params.grad *= (coeff[i] * cfg.rou)

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            optimizer.step() 
            pbar.update(1)
        
        scheduler.step()

    return total_loss/len(dataloader),total_loss_1/len(dataloader),total_loss_2/len(dataloader)


def val(model,device,dataloader):
    softmax=nn.Softmax(dim=1)
    sum_all=0
    sum_1=0
    sum_2=0
    tot=0
    all_out=[]
    all_label=[]
    with torch.no_grad():
        model.eval()
        for step,(spec,img,label) in enumerate(dataloader):
            spec=spec.to(device)
            img=img.to(device)
            label=label.to(device)
            out_1,out_2,out,update_flag,performance_1,performance_2=model(spec.unsqueeze(1).float(),img.float(),label,warm_up=2)
            prediction=softmax(out)
            pred_1=softmax(out_1)
            pred_2=softmax(out_2)
            tot+=img.shape[0]
            sum_all+=torch.sum(torch.argmax(prediction,dim=1)==label).item()
            sum_1+=torch.sum(torch.argmax(pred_1,dim=1)==label).item()
            sum_2+=torch.sum(torch.argmax(pred_2,dim=1)==label).item()
            
            for i in range(label.shape[0]):
                all_out.append(prediction[i].cpu().data.numpy())
                ss=torch.zeros(31)
                ss[label[i]]=1
                all_label.append(ss.numpy())
        
        all_out=np.array(all_out)
        all_label=np.array(all_label)
        mAP=average_precision_score(all_label,all_out)
        

    return mAP,sum_all/tot,sum_1/tot,sum_2/tot


def write2txt(fp,info,mode='a'):
    with open(fp,mode=mode) as f:
        f.write(info)
        f.write('\n')


def main():
    # job_id=datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    cfg = Config()
    args=get_arguments()
    cfg.parse(vars(args))
    setup_seed(cfg.random_seed)

    job_name=args.exp_name
    cur_dir=os.path.join('results',job_name)
    os.makedirs(cur_dir,exist_ok=True)
    
    # log=Logger(os.path.join(cur_dir,'log.log'),level='info')
    writer=None
    if cfg.use_tensorboard:
        writer_path=os.path.join(cur_dir,'tensorboard')
        os.makedirs(writer_path,exist_ok=True)
        writer=SummaryWriter(writer_path)

    saved_data=vars(cfg)
    cmd=' '.join(sys.argv)
    saved_data.update({'cmd':cmd})
    saved_data=json.dumps(saved_data,indent=4)
    with open(os.path.join(cur_dir,'config.json'),'w') as f:
        f.write(saved_data)
    
    device=torch.device('cuda')

    spatial_transforms=get_spatial_transform(opt=cfg)
    val_spatial_transforms=get_val_spatial_transforms(opt=cfg)
#     train_dataset=KSDataset(mode='training',spatial_transform=spatial_transforms,video_loader=VideoLoaderHDF5())
#     test_dataset=KSDataset(mode='testing',spatial_transform=val_spatial_transforms,video_loader=VideoLoaderHDF5(),audio_drop=cfg.audio_drop,visual_drop=cfg.visual_drop)
    
    train_dataset = CramedDataset(args, mode='train')
    test_dataset = CramedDataset(args, mode='test')

    train_loader=DataLoader(train_dataset,batch_size=cfg.batch_size,shuffle=True,num_workers=32,pin_memory=True)
    test_loader=DataLoader(test_dataset,batch_size=cfg.batch_size,shuffle=False,num_workers=32,pin_memory=True)

    cfg.num_mod = 2
    cfg.l_gm = None
    cfg.pre_perform = [0] * cfg.num_mod
    
    model=Classifier(cfg,device=device)

    if cfg.resume_model:
        state_dict=torch.load(cfg.resume_model_path,map_location='cuda')
        model.load_state_dict(state_dict=state_dict)
    else:
        model.apply(weight_init)
    
    model.to(device)

    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=0.01)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=cfg.lr_decay_step,gamma=cfg.lr_decay_ratio)

    start_epoch=-1
    best_acc=0.0
    logger_path=join(cur_dir,'log.txt')

    if cfg.train:
        for epoch in range(start_epoch+1,cfg.epochs):
            loss,loss_1,loss_2=train(cfg,epoch,model,device,train_loader,optimizer,scheduler,tb=writer)
            mAP,acc,acc_1,acc_2=val(model,device,test_loader)
            # log.logger.info('epoch:{} acc:{:.4f} acc_1:{:.4f} acc_2:{:.4f} mAP:{:.4f}'.format(epoch,acc,acc_1,acc_2,mAP))
            write2txt(fp=logger_path,info=f'epoch:{epoch} acc:{acc:.4f} acc_1:{acc_1:.4f} acc_2:{acc_2:.4f} mAP:{mAP:.4f}')
            if writer is not None:
                writer.add_scalars(main_tag='Loss',tag_scalar_dict={'loss':loss,'loss_1':loss_1,'loss_2':loss_2},global_step=epoch)
                writer.add_scalars(main_tag='Acc',tag_scalar_dict={'acc':acc,'acc_1':acc_1,'acc_2':acc_2},global_step=epoch)

            if acc>best_acc:
                best_acc=acc
                saved_data={}
                saved_data['epoch']=epoch
                saved_data['acc']=acc
                saved_data['mAP']=mAP
                saved_data['acc_1']=acc_1
                saved_data['acc_2']=acc_2
                saved_data=json.dumps(saved_data,indent=4)

                with open(os.path.join(cur_dir,'best_model.json'),'w') as f:
                    f.write(saved_data)

                torch.save(model.state_dict(),os.path.join(cur_dir,'best_model.pth'))
    else:
        mAP,acc,acc_1,acc_2=val(model,device,test_loader)
        print('mAP:{} Acc:{}'.format(mAP,acc))
    

if __name__ == "__main__":
    main()
