from Plugins import dataset
from Plugins import Config
from network.unet import unet
from network.attention_unet import attention_unet
import torch
from Plugins.ImageQuality import PSNR
from Plugins.logger import Logger
from torch.optim import lr_scheduler

def pre_train():
    args = Config.getArgparse()
    sp = Config.getScanParam()

    train_loader = dataset.image_loader(args.input_path,sp['nWidth'],
                                             batch_size=args.batch_size,shuffle=True)
    test_loader = dataset.image_loader(args.test_path,image_size=sp['nWidth'],
                                            batch_size=1,shuffle=False)
    val_loader = dataset.image_loader(args.input_path,sp['nWidth'],
                                             batch_size=1,shuffle=False)

    if args.model == 'unet':
        model = unet(1,1).cuda()
        logs = Logger(r'./logger/unet_'+args.is_noise+'.txt')
    elif args.model == 'attention_unet':
        model = attention_unet(1,1,args.k1,args.k2).cuda()
        logs = Logger(r'./logger/attention_unet_'+args.is_noise+'_'+str(args.k1)+'_'+str(args.k2)+'.txt')
    else:
        raise NotImplementedError

    loss = torch.nn.MSELoss()
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=args.lr)
    # scheduer = lr_scheduler.MultiStepLR(opt,[i*args.epoch//5 for i in range(5)], gamma=0.5)
    scheduer = lr_scheduler.StepLR(opt,gamma=0.5,step_size=args.epoch//5)
    for epoch in range(args.epoch):
        model.train()
        for step,(x,y,file_name) in enumerate(train_loader):
            out = model(x.cuda())
            loss_value = loss(out,y.cuda())
            opt.zero_grad()
            loss_value.backward()
            opt.step()
            if (step%400) == 0:
                psnr_ = PSNR(out.detach(), y.cuda()).cpu()
                logs.append('epoch:{},step:{},loss:{},psnr:{}'.format(epoch,step,loss_value,psnr_))
                out.data.cpu().numpy().tofile('./temp_res/' + args.model + '_res.raw')
                y.data.cpu().numpy().tofile('./temp_res/' + args.model + '_lab.raw')
        scheduer.step()
        model.eval()
        if args.model == 'unet':
            torch.save(model,r'./checkpoints/unet_' + args.is_noise + '.pt')
        elif args.model == 'attention_unet':
            torch.save(model,r'./checkpoints/attention_unet_' + args.is_noise + '_' + str(args.k1) + '_' + str(args.k2) + '.pt')
        else:
            raise NotImplementedError

    model.eval()


if __name__ == "__main__":
    pre_train()

