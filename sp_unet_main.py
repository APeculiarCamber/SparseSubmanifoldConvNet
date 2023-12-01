import sp_data as data
import torch.nn as nn
import torch
import torch.optim as optim
import os
import numpy as np
import sys
import math, time



data.init(-1,24,24*8,16)





class ConvBlock(nn.Module):
    """Some Information about ConvBlock"""
    def __init__(self, in_features, out_features, reps):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm3d(in_features),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_features, out_features, 3, 1, 1),
        )
        for i in range(reps):
            self.conv.append(nn.BatchNorm3d(out_features))
            self.conv.append(nn.LeakyReLU(out_features))
            self.conv.append(nn.Conv3d(out_features, out_features, 3, 1, 1))
        
    def forward(self, x):
        return self.conv(x)

class DownConv(nn.Module):
    """Some Information about DownConv"""
    def __init__(self, in_features, out_features):
        super(DownConv, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm3d(in_features),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_features, out_features, 2, 2, 0),
        )

    def forward(self, x):
        return self.conv(x)

class UpConv(nn.Module):
    """Some Information about DownConv"""
    def __init__(self, in_features, out_features):
        super(UpConv, self, ).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm3d(in_features),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose3d(in_features, out_features, 4, 2, 1),
        )
    
    def forward(self, x):
        return self.conv(x)


class BaseUNET(nn.Module):
    def __init__(self, reps, nPlanes, n_grid, num_classes, downsampling=2):
        super(BaseUNET, self, ).__init__()

        self.num_classes = num_classes
        self.n_grid = n_grid

        self.start_conv = ConvBlock(1, nPlanes[0], 0)

        nn.ModuleList
        self.down_conv_blocks = nn.ModuleList()
        for n in nPlanes[:-1]:
            self.down_conv_blocks.append(ConvBlock(n, n, reps))
        self.downsampling_convs = nn.ModuleList()
        for up,down in zip(nPlanes[:-1], nPlanes[1:]):
            self.downsampling_convs.append(DownConv(up, down))


        self.mid_conv_block = ConvBlock(nPlanes[-1], nPlanes[-1], reps)
        self.mid_up_block = UpConv(nPlanes[-1], nPlanes[-2])

        self.up_conv_blocks = nn.ModuleList()
        for n in reversed(nPlanes[1:-1]):
            self.up_conv_blocks.append(ConvBlock(n+n, n, reps))

        self.upsampling_convs = nn.ModuleList()
        for down,up in zip(reversed(nPlanes[1:-1]), reversed(nPlanes[:-2])):
            self.upsampling_convs.append(UpConv(down, up))


        self.linear_per_pixel = nn.Sequential(
            nn.Linear(nPlanes[0]*2 + 3, nPlanes[0]),
            nn.LeakyReLU(inplace=True),
            nn.Linear(nPlanes[0], num_classes),
            nn.ReLU(inplace=True),
            nn.Softmax(dim=-1),
        )

        self.debug = True

    def forward(self, x):
        '''
        TODO : revise dataloader
        assume [-1.0, 1.0], its normalized by max

        x_pos += 0.5
        x_pos *= n_grid
        x_inds = torch.floor(x_pos).clamp(min=0, max=n_grid-1)
        x_pos = x_inds
        x_inds = x_inds.to(torch.long)
        '''

        with torch.no_grad():
            x_pos = x[0].cuda()
            x_pos = ((x_pos + 1.0) * (self.n_grid / 2))
            oob = torch.any(torch.logical_and(x_pos >= self.n_grid-1, x_pos < 0))
            if oob: print("OOB")
            else: print("GOOD")
            x_inds = torch.floor(x_pos).clamp(min=0, max=self.n_grid-1)
            x_pos -= x_inds
            x_inds = x_inds.to(torch.long)
        
        grid = torch.zeros(x_inds[-1,3] + 1, 1, self.n_grid, self.n_grid, self.n_grid, device=x_inds.device)
        grid[x_inds[:,3], :, x_inds[:,0], x_inds[:,1], x_inds[:,2]] = 1.0

        if self.debug: print("Before start:", grid.shape)
        x = self.start_conv(grid)
        if self.debug: print("After start:", x.shape)

        # DOWN
        skips = []
        for i, (conv_block, down_block) in enumerate(zip(self.down_conv_blocks, self.downsampling_convs)):
            x = conv_block(x)
            skips.append(x)
            x = down_block(x)
            if self.debug: print(f"After down conv {i}:", x.shape)
        
        # MID
        x = self.mid_conv_block(x)
        x = self.mid_up_block(x)
        if self.debug: print(f"After mid conv:", x.shape)

        # UP
        for i, (conv_block, up_block) in enumerate(zip(self.up_conv_blocks, self.upsampling_convs)):
            skip = skips[-1 - i]
            if self.debug: print(f"Before up conv {i}:", x.shape, skips[-1 - i].shape)
            x = torch.cat([x, skip], dim=1)
            x = conv_block(x)
            x = up_block(x)
            if self.debug: print(f"After up conv {i}:", x.shape)

        # LINEAR : 
        # TODO, two options: ASSIGN DIRECT, or split and append
        x = torch.cat([x, skips[0]], dim=1)
        if self.debug: print(f"After CONVs", x.shape)

        x_per_pixel = x[x_inds[:,3], :, x_inds[:,0], x_inds[:,1], x_inds[:,2]]
        x_per_pixel = torch.cat([x_per_pixel, x_pos], dim=-1)
        print("WINNING", x_per_pixel.shape)

        return self.linear_per_pixel(x_per_pixel)

dimension = 3
reps = 1 #Conv block repetition factor
m = 16 #Unet number of features
nPlanes = [m, 2*m, 3*m, 4*m, 5*m] # UNet number of features per level
# each convolutions preceded by batch normalization and a ReLU non-linearity
nGrid = 64
model=BaseUNET(reps, nPlanes, nGrid, data.nClassesTotal)
print(model)
trainIterator=data.train()
validIterator=data.valid()

criterion = nn.CrossEntropyLoss()
p={}
p['n_epochs'] = 100
p['initial_lr'] = 1e-1
p['lr_decay'] = 4e-2
p['weight_decay'] = 1e-4
p['momentum'] = 0.9
p['check_point'] = False
p['use_cuda'] = torch.cuda.is_available()
dtype = 'torch.cuda.FloatTensor' if p['use_cuda'] else 'torch.FloatTensor'
dtypei = 'torch.cuda.LongTensor' if p['use_cuda'] else 'torch.LongTensor'


if p['use_cuda']:
    model.cuda()
    criterion.cuda()
optimizer = optim.SGD(model.parameters(),
    lr=p['initial_lr'],
    momentum = p['momentum'],
    weight_decay = p['weight_decay'],
    nesterov=True)
if p['check_point'] and os.path.isfile('epoch.pth'):
    p['epoch'] = torch.load('epoch.pth') + 1
    print('Restarting at epoch ' +
          str(p['epoch']) +
          ' from model.pth ..')
    model.load_state_dict(torch.load('model.pth'))
else:
    p['epoch']=1
print(p)
print('#parameters', sum([x.nelement() for x in model.parameters()]))




def store(stats,batch,predictions,loss):
    ctr=0
    for nP,f,classOffset,nClasses in zip(batch['nPoints'],batch['xf'],batch['classOffset'],batch['nClasses']):
        categ,f=f.split('/')[-2:]
        if not categ in stats:
            stats[categ]={}
        if not f in stats[categ]:
            stats[categ][f]={'p': 0, 'y': 0}
        #print(predictions[ctr:ctr+nP,classOffset:classOffset+nClasses].abs().max().item())
        stats[categ][f]['p']+=predictions.detach()[ctr:ctr+nP,classOffset:classOffset+nClasses].cpu().numpy()
        stats[categ][f]['y']=batch['y'].detach()[ctr:ctr+nP].cpu().numpy()-classOffset
        ctr+=nP

def inter(pred, gt, label):
    assert pred.size == gt.size, 'Predictions incomplete!'
    return np.sum(np.logical_and(pred.astype('int') == label, gt.astype('int') == label))

def union(pred, gt, label):
    assert pred.size == gt.size, 'Predictions incomplete!'
    return np.sum(np.logical_or(pred.astype('int') == label, gt.astype('int') == label))

def iou(stats):
    eps = sys.float_info.epsilon
    categories= sorted(stats.keys())
    ncategory = len(categories)
    iou_all = np.zeros(ncategory)
    nmodels = np.zeros(ncategory, dtype='int')
    for i, categ in enumerate(categories):
        nmodels[i] = len(stats[categ])
        pred = []
        gt = []
        for j in stats[categ].values():
            pred.append(j['p'].argmax(1))
            gt.append(j['y'])
        npart = np.max(np.concatenate(gt))+1
        iou_per_part = np.zeros((len(pred), npart))
        # loop over parts
        for j in range(npart):
            # loop over CAD models
            for k in range(len(pred)):
                p = pred[k]
                iou_per_part[k, j] = (inter(p, gt[k], j) + eps) / (union(p, gt[k], j) + eps)
        # average over CAD models and parts
        iou_all[i] = np.mean(iou_per_part)
    # weighted average over categories
    iou_weighted_ave = np.sum(iou_all * nmodels) / np.sum(nmodels)
    return {'iou': iou_weighted_ave, 'nmodels_sum': nmodels.sum(), 'iou_all': iou_all}




for epoch in range(p['epoch'], p['n_epochs'] + 1):
    model.train()
    stats = {}
    for param_group in optimizer.param_groups:
        param_group['lr'] = p['initial_lr'] * \
        math.exp((1 - epoch) * p['lr_decay'])
    start = time.time()
    for batch in trainIterator:
        optimizer.zero_grad()
        batch['x'][1]=batch['x'][1].type(dtype)
        batch['y']=batch['y'].type(dtypei)
        batch['mask']=batch['mask'].type(dtype)
        predictions=model(batch['x'])
        loss = criterion.forward(predictions,batch['y'])
        store(stats,batch,predictions,loss)
        loss.backward()
        optimizer.step()        
    r = iou(stats)
    print('train epoch',epoch,1,'iou=', r['iou'], 'MegaMulAdd=','time=',time.time() - start,'s')

    if p['check_point']:
        torch.save(epoch, 'epoch.pth')
        torch.save(model.state_dict(),'model.pth')

    if epoch in [10,30,100]:
        model.eval()
        stats = {}
        start = time.time()
        for rep in range(1,1+3):
            for batch in validIterator:
                batch['x'][1]=batch['x'][1].type(dtype)
                batch['y']=batch['y'].type(dtypei)
                batch['mask']=batch['mask'].type(dtype)
                predictions=model(batch['x'])
                loss = criterion.forward(predictions,batch['y'])
                store(stats,batch,predictions,loss)
            r = iou(stats)
            print('valid epoch',epoch,rep,'iou=', r['iou'],'time=',time.time() - start,'s')
        print(r['iou_all'])
