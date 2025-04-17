
import warnings
warnings.filterwarnings('ignore')
import argparse
import torch
import torch.optim as optim
import numpy as np
import time
import random
import json
import os
from os.path import join
import sys
from tqdm import tqdm
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.dataloader import MosiDataLoader
# from models.Classifier import Classifier
from models.msamodel import MSAModel

from config import Config
# from dataset.KS import KSDataset
from dataset.CramedDataset import CramedDataset
# from utils.log_file import Logger
from datetime import datetime

from dataset.spatial_transforms import get_spatial_transform,get_val_spatial_transforms
from dataset.loader import VideoLoaderHDF5

from sklearn.metrics import average_precision_score
from src.eval_metrics import *
import torch.nn.functional as F

TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now()) 


def get_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', default='mosi', type=str,
                        help='sarcasm, mosei, CREMAD')
    # parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--data_path', type=str, default='/kaggle/input/dataset-mosi/mosi_data.pkl')
    # parser.add_argument('--audio_path', default='/kaggle/input/cremad/AudioWAV/AudioWAV', type=str)
    # parser.add_argument('--visual_path', default='/kaggle/input/cremad/Image-01-FPS', type=str)
    # parser.add_argument('--use_modulation',action='store_true',help='use gradient modulation')
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
    parser.add_argument('--cls_lr', type=float, default=5e-7,
                        help='classifier learning rate')
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--epochs', type=int, default=60,
                        help='number of epochs')
    parser.add_argument('--when', type=int, default=10,
                        help='when to decay learning rate')
    parser.add_argument('--rou', type=float, default=2.3) 
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

def train(cfg,epoch,model,device,dataloader,optimizer,scheduler,tb=None, criterion=None):
    # global pre_perform, l_gm
    loss_fn=nn.CrossEntropyLoss().to(device)
    # relu=nn.ReLU(inplace=True)
    # tanh=nn.Tanh()
    model.train()
    total_loss=0
    total_loss_1=0
    total_loss_2=0
    total_loss_3=0
    with tqdm(total=len(dataloader), desc=f"Train-epoch-{epoch}") as pbar:
        for step, batch in  enumerate(dataloader):
            text=batch['text'].to(device) # b,h,w
            audio=batch['audio'].to(device) 
            vision=batch['vision'].to(device)# b,c,t,h,w
            label=batch['labels'].squeeze(-1).to(device)# If num of labels is 1 as [batch, 1], remove 1
            # print(f"Label is: {label}") # Ex: MOSI:[[1.23],...]; MOSEI:[[1.123343, 0.000, 0.000, 0.000, 0.00000, 0.000, 0.0000],...,]
            # print (f"Label shape is:{label.shape}") # [64, 1, 7] = [batch_size, seq_len, channel/dim]
            
            optimizer.zero_grad()
            batch_size = text.size(0)
            # net = nn.DataParallel(model) if batch_size > 10 else model
            warm_up=1 if epoch<=5 else 0
            # warm_up=0
            out_1,out_2,out_3,out,update_flag,performance_1,performance_2,performance_3,cls_grad,cls_optimizer = model([text, audio, vision],label,warm_up,use_grad=True)
            # print(f"STEP: {step}")
            # print(f"update flag after use model at step {step}: {update_flag.shape}")
            
            if cfg.use_adam_drop:
                if torch.all(torch.sum(update_flag, dim=0) == 0):
                    continue
                select_mask=update_flag!=0 # This is to keep just features != 0 *non-dropped features.
                # print(f"Select mask(update_flag after remove dropped batches) at step {step}: {select_mask.shape}")
                
                label=label[select_mask]
                out_1=out_1[select_mask]
                out_2=out_2[select_mask]
                out_3=out_3[select_mask]
                # print(f"label after dropped: {label.shape}")
                # print(f"out total modal after dropped: {out.shape}")
                # print(f"out_1 text after dropped: {out_1.shape}")
                # print(f"out_2 audio after dropped: {out_2.shape}")
                # print(f"out_3 vision after dropped: {out_3.shape}")
            print("=" * 70)

            loss=criterion(out,label) # sum of [text, audio vision]
            loss_1=criterion(out_1,label) # text
            loss_2=criterion(out_3, label)# audio
            loss_3=criterion(out_2,label) # vision
            total_loss+=loss.item()
            total_loss_1+=loss_1.item()
            total_loss_2+=loss_2.item()
            total_loss_3+=loss_3.item()

            # Gradient Magnitude Modulation
            if cfg.l_gm is not None:
                loss += cfg.lamda * cfg.l_gm  
            loss.backward()

            # Get the gradient of fusion module
            for name, para in model.named_parameters():
                if 'out_layer.weight' in name:
                    fusion_grad = para

            llist = cal_cos(cls_grad, fusion_grad) # Calculate sim(classifiers_grad, fusion_grad) using cosine similarity
            
            diff = [performance_1-cfg.pre_perform[0], performance_2-cfg.pre_perform[1], performance_3-cfg.pre_perform[2]] # text, audio, vision
            
            diff_sum = sum(diff) + 1e-8
            coeff = list()
            # Calculate gradient magnitude Bt
            for d in diff:
                coeff.append((diff_sum - d) / diff_sum)
                
            # Update performance previous step and l_gm
            cfg.pre_perform[0] = performance_1
            cfg.pre_perform[1] = performance_2
            cfg.pre_perform[2] = performance_3
            cfg.l_gm = np.sum(np.abs(coeff)) - (coeff[0] * llist[0] + coeff[1] * llist[1] + coeff[2] * llist[2])
            cfg.l_gm /= cfg.num_mod

            # Update parameters of each modality θϕi
            for i in range(cfg.num_mod):
                for name, params in model.named_parameters():
                    if f'encoders.{i}' in name:
                        params.grad *= (coeff[i] * cfg.rou)
            cls_optimizer.step()

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            optimizer.step() 
            pbar.update(1)
        
        # scheduler.step()
    # return for mode "test"
    return total_loss/len(dataloader),total_loss_1/len(dataloader),total_loss_2/len(dataloader), total_loss_3/len(dataloader)


def val(model,device,dataloader,criterion=None, cfg=None, mode='test'):
    # softmax=nn.Softmax(dim=1)
    loss_all=0
    loss_1=0
    loss_2=0
    loss_3=0
    total=cfg.n_valid if mode=='valid' else cfg.n_test
    # For test dataset
    results = []
    truths = []
    # For valid dataset
    val_results=[]
    val_labels=[]
    
    with torch.no_grad():
        model.eval()
        for step, batch in enumerate(dataloader):
            text=batch['text'].to(device)
            audio=batch['audio'].to(device)
            vision=batch['vision'].to(device)
            label=batch['labels'].squeeze(-1).to(device)

            net = model
            out_1,out_2,out_3,out,update_flag,performance_1,performance_2, performance_3=net([text, audio, vision],label,warm_up=1,use_grad=False)
            
            # select_mask=update_flag!=0 # This is to keep just features != 0 *non-dropped features.
            # print(f"Select mask(update_flag after remove dropped batches) at step {step}: {select_mask.shape}")
            # label=label[select_mask]
            # out_1=out_1[select_mask]
            # out_2=out_2[select_mask]
            # out_3=out_3[select_mask]
            
            loss_all += criterion(out, label).item()
            loss_1 += criterion(out_1, label).item()
            loss_2 += criterion(out_2, label).item()
            loss_3 += criterion(out_3, label).item()
            
            if mode=='valid':
                # val_results=[]
                # val_labels=[]
                val_results.append(out)
                val_labels.append(label)
                return loss_all/total, val_results, val_labels
            else:    
                results.append(out)
                truths.append(label)
        # all_out=np.array(all_out)
        # all_label=np.array(all_label)
        # mAP=average_precision_score(all_label,all_out)
        

    return loss_all/total, loss_1/total, loss_2/total, loss_3/total, results,truths


def write2txt(fp,info,mode='a'):
    with open(fp,mode=mode) as f:
        f.write(info)
        f.write('\n')


def main():
    # job_id=datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    cfg = Config()
    args=get_arguments()
    cfg.parse(vars(args))
    dataset = str.lower(args.dataset.strip())
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

    dataloader, orig_dim = MosiDataLoader(args.dataset, args.batch_size, args.data_path)
    train_loader = dataloader['train']
    valid_loader = dataloader['valid']
    test_loader = dataloader['test']

    cfg.orig_dim = orig_dim
    cfg.layers = args.nlevels
    cfg.when = args.when
    cfg.n_train, cfg.n_valid, cfg.n_test = len(train_loader), len(valid_loader), len(test_loader)
    cfg.output_dim = 1
    cfg.criterion = 'L1Loss'
    cfg.num_mod = 3

    cfg.l_gm=None
    cfg.pre_perform = [0] * cfg.num_mod
    
    # model=Classifier(cfg,device=device)
    model = MSAModel(cfg.output_dim, cfg.orig_dim, cfg.proj_dim,
                 cfg.num_heads, cfg.layers, cfg.relu_dropout,
                 cfg.embed_dropout, cfg.res_dropout, cfg.out_dropout,
                 cfg.attn_dropout, cfg.cls_layers, cfg.q_base, cfg.p_exe, 
                 cfg.d, cfg.lam, cfg.optim, cfg.cls_lr, device=device)

    if cfg.resume_model:
        state_dict=torch.load(cfg.resume_model_path,map_location='cuda')
        model.load_state_dict(state_dict=state_dict)
    else:
        model.apply(weight_init)
    
    model.to(device)

    # optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=0.01)
    # scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=cfg.lr_decay_step,gamma=cfg.lr_decay_ratio)
    optimizer = getattr(optim, cfg.optim)(model.parameters(), lr=cfg.lr)
    
    criterion = getattr(nn, cfg.criterion)()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=cfg.when, factor=0.1, verbose=True)
    # settings = {'model': model, 'optimizer': optimizer, 'criterion': criterion, 'scheduler': scheduler,
    #             'classifier': classifier, 'cls_optimizer': cls_optimizer}
    
    start_epoch=-1
    best_acc=0.0
    logger_path=join(cur_dir,'log.txt')

    if cfg.train:
        for epoch in range(start_epoch+1,cfg.epochs):
            loss,loss_1,loss_2,loss_3=train(cfg,epoch,model,device,train_loader,optimizer,scheduler,tb=writer, criterion=criterion)
            val_loss,results,truths=val(model,device,valid_loader,criterion=criterion, cfg=cfg, mode='valid') #Important: Here acc is average validation loss
            
            scheduler.step(val_loss) # Decay learning rate by validation loss
            
            # Calculate accuracy
            results = torch.cat(results)
            truths = torch.cat(truths)
            acc = train_eval_senti(results, truths) # Get the accuracy
            
            write2txt(fp=logger_path,info=f'epoch:{epoch} acc:{acc:.4f}')
            if writer is not None:
                writer.add_scalars(main_tag='Loss',tag_scalar_dict={'loss':loss},global_step=epoch)
                writer.add_scalars(main_tag='Acc',tag_scalar_dict={'acc':acc},global_step=epoch)

            # Use val_loss instead of accuracy
            if acc>best_acc:
                best_acc=acc
                saved_data={}
                saved_data['epoch']=epoch
                saved_data['acc']=acc
                # saved_data['acc_1']=acc_1
                # saved_data['acc_2']=acc_2
                # saved_data['acc_3']=acc_3
                saved_data=json.dumps(saved_data,indent=4)

                with open(os.path.join(cur_dir,'best_model.json'),'w') as f:
                    f.write(saved_data)

                torch.save(model.state_dict(),os.path.join(cur_dir,'best_model.pth'))
    else:
        loss,acc_1,acc_2,acc_3,results,truths=val(model,device,test_loader, criterion, cfg, mode='test')
        results = torch.cat(results)
        truths = torch.cat(truths)
        acc, f_score = eval_senti(results, truths)
        print(f'Loss: {loss}, Acc: {acc}, F1_score: {f_score}')
    

if __name__ == "__main__":
    main()
