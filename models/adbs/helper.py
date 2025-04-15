# import new Network name here and add in model_class args
from .Network import MYNET
from utils import *
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F

def new_train(trainloader, model, session, args):
    old_class = args.base_class + args.way * (session - 1)
    new_class = args.base_class + args.way * session
    new_margin = nn.Parameter(torch.rand(args.way, device="cuda", requires_grad=True))
    new_margin.data.copy_(model.margin[old_class : new_class].data)
    
    optimizer = torch.optim.SGD([{'params': new_margin, 'lr': args.lr_new}],
                                momentum=0.9, dampening=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones2,
                                                     gamma=args.gamma2)

    with torch.enable_grad():
        for epoch in range(args.epochs_new_train):
            for batch in trainloader:
                data, labels = [_ for _ in batch]
                b, c, h, w = data[1].shape
                data = data[0].cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

            old_margin = model.margin[:old_class].clone().detach()
            margin = torch.cat([old_margin, new_margin], dim=0)
            features, _ = model.encode(data)
            features.detach()
            logits = model.get_logits(features, model.fc.weight[:new_class, :].clone().detach())
            logits = logits * margin

            cls_loss = F.cross_entropy(logits, labels)
            
            if args.margin:
                # (1 - w_i*w_j) - (m_i - m_j*w_i*w_j) <= 0
                diff_matrx = diff_loss_w(margin, F.normalize(model.fc.weight[:new_class, :], p=2, dim=-1))
                reg_loss = args.reg_alpha * torch.mean(torch.relu(diff_matrx))
            
            loss = cls_loss + reg_loss

            tl = loss.item()
            cl = cls_loss.item()
            rl = reg_loss.item()
            lrc = scheduler.get_last_lr()[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.margin.data[old_class: new_class].copy_(torch.tensor(new_margin, device="cuda"))

            scheduler.step()

def diff_loss_w(m, w):
    # Calculate IC loss
    n = w.size(0)
    w_dot_product = torch.mm(w, w.t())  # w_i * w_j for all pairs
    m_expanded = m.expand(n, n)
    m_i_minus_mj_wiwj = m_expanded - m_expanded.t() * w_dot_product
    diff_matrix = (1 - w_dot_product) - m_i_minus_mj_wiwj
    return diff_matrix

def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    tl_cls = Averager()
    tl_reg = Averager() 
    ta = Averager()
    model = model.train()
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        data, labels = [_ for _ in batch]
        b, c, h, w = data[1].shape
        data = data[0].cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        
        preds = model(im_cla=data) 
        
        preds = preds[:, :args.base_class]
        cls_loss = F.cross_entropy(preds, labels)

        if args.margin:
            margin = model.margin[:args.base_class]
            # (1 - w_i*w_j) - (m_i - m_j*w_i*w_j) <= 0
            diff_matrx = diff_loss_w(margin, F.normalize(model.fc.weight[:args.base_class, :], p=2, dim=-1))
            reg_loss = args.reg_alpha * torch.mean(torch.relu(diff_matrx))

        loss = cls_loss + reg_loss 
        total_loss = loss
        
        acc = count_acc(preds, labels)

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        tl_cls.add(cls_loss.item())
        if args.margin:
            tl_reg.add(reg_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    tl_cls = tl_cls.item()
    tl_reg = tl_reg.item()
    return tl, tl_cls, ta, tl_reg


def replace_base_fc(trainset, test_transform, data_transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=64,
                                              num_workers=8, pin_memory=True, shuffle=False) 
    trainloader.dataset.transform = test_transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            b = data.size()[0]
            data = data_transform(data)
            m = data.size()[0] // b
            labels = torch.stack([label*m+ii for ii in range(m)], 1).view(-1)
            model.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(labels.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class*m):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.fc.weight.data[:args.base_class*m] = proto_list

    return model

def test(model, testloader, epoch, args, session):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            b = data.size()[0]
            preds = model(data)
            preds = preds[:, :test_class]

            loss = F.cross_entropy(preds, test_label)
            acc = count_acc(preds, test_label)

            vl.add(loss.item())
            va.add(acc)

        vl = vl.item()
        va = va.item()
    print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

    return vl,va