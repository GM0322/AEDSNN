from Plugins import dataset
from Plugins import Config
from network.unet import unet
from network.attention_unet import aedsnn
import torch
from Plugins.ImageQuality import PSNR
from Plugins.logger import Logger
from torch.optim import lr_scheduler

def train():
    args = Config.getArgparse()
    sp = Config.getScanParam()

    train_loader = dataset.aeds_loader(args.proj_path,sp['nViews'],sp['nBins'],sp['nWidth'], batch_size=args.batch_size,shuffle=True)

    model = aedsnn(args.k1,args.k2,args.n).cuda()
    logs = Logger(r'./logger/aedsnn'+args.is_noise+'_'+str(args.k1)+'_'+str(args.k2)+'.txt')

    loss = torch.nn.MSELoss()
    opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr=args.lr)
    # scheduer = lr_scheduler.MultiStepLR(opt,[i*args.epoch//5 for i in range(5)], gamma=0.5)
    scheduer = lr_scheduler.StepLR(opt,gamma=0.5,step_size=args.epoch//5)
    for epoch in range(args.epoch):
        model.train()
        for step,(p,y,file_name) in enumerate(train_loader):
            out = model(torch.zeros_like(y,dtype=torch.float32).cuda(),p.cuda())
            loss_value = loss(out,y.cuda())
            opt.zero_grad()
            loss_value.backward()
            opt.step()
            if (step%10) == 0:
                psnr_ = PSNR(out.detach(), y.cuda()).cpu()
                logs.append('epoch:{},step:{},loss:{},psnr:{}'.format(epoch,step,loss_value,psnr_))
                out.data.cpu().numpy().tofile('./temp_res/res.raw')
                y.data.cpu().numpy().tofile('./temp_res/lab.raw')
        scheduer.step()

if __name__ == "__main__":
    train()
