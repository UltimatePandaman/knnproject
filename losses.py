#coding:utf-8

import os
import torch

from torch import nn
from munch import Munch
from transforms import build_transforms

import torch.nn.functional as F
import numpy as np

def compute_d_loss(nets, args, x_real_a, x_real_b, use_r1_reg=True, use_adv_cls=False, use_con_reg=False):
    args = Munch(args)
    
    ### REAL AUDIOS ###
    # with real audios
    x_real_a.requires_grad_()
    x_real_b.requires_grad_()
    y = torch.zeros(x_real_a.size(0))
    out_a = nets.discriminator(x_real_a, torch.zeros(x_real_a.size(0)))
    out_b = nets.discriminator(x_real_b, torch.ones(x_real_b.size(0)))
    loss_real = adv_loss(out_a, 1)
    loss_real += adv_loss(out_b, 1)
    
    # R1 regularizaition (https://arxiv.org/abs/1801.04406v4)
    if use_r1_reg:
        loss_reg = r1_reg(out_a, x_real_a)
        loss_reg += r1_reg(out_b, x_real_b)
    else:
        loss_reg = torch.FloatTensor([0]).to(x_real_a.device)
    
    # consistency regularization (bCR-GAN: https://arxiv.org/abs/2002.04724)
    loss_con_reg = torch.FloatTensor([0]).to(x_real_a.device)
    if use_con_reg:
        t = build_transforms()
        out_aug_a = nets.discriminator(t(x_real_a).detach(), torch.zeros(x_real_a.size(0)))
        out_aug_b = nets.discriminator(t(x_real_b).detach(), torch.ones(x_real_b.size(0)))
        loss_con_reg += F.smooth_l1_loss(out_a, out_aug_a)
        loss_con_reg += F.smooth_l1_loss(out_b, out_aug_b)
    
    ### FAKE AUDIOS ###
    # with fake audios
    with torch.no_grad():
        F0_a = nets.f0_model.get_feature_GAN(x_real_a)
        x_fake_b = nets.generator_b(x_real_a, masks=None, F0=F0_a)
        x_recon_a = nets.generator_a(x_fake_b, masks=None, F0=F0_a) # I use a twice because I don't know if F0 is being fine-tuned to handle generated output
        
        F0_b = nets.f0_model.get_feature_GAN(x_real_b)
        x_fake_a = nets.generator_a(x_real_b, masks=None, F0=F0_b)
        x_recon_b = nets.generator_b(x_fake_a, masks=None, F0=F0_b)
        
    # Fake audios loss
    out_a = nets.discriminator(x_fake_a, torch.zeros(x_fake_a.size(0)))
    out_b = nets.discriminator(x_fake_b, torch.ones(x_fake_b.size(0)))
    loss_fake = adv_loss(out_a, 0)
    loss_fake += adv_loss(out_b, 0)
    if use_con_reg:
        out_aug_a = nets.discriminator(t(x_fake_a).detach(), torch.zeros(x_fake_a.size(0)))
        out_aug_b = nets.discriminator(t(x_fake_b).detach(), torch.ones(x_fake_b.size(0)))
        loss_con_reg += F.smooth_l1_loss(out_a, out_aug_a)
        loss_con_reg += F.smooth_l1_loss(out_b, out_aug_b)
    
    # adversarial classifier loss
    ## isn't it stated in the apper that the classifier learns on real samples?
    if use_adv_cls:
        out_de_a = nets.discriminator.classifier(x_fake_a)
        out_de_b = nets.discriminator.classifier(x_fake_b)
        ## here the ones and zeros are swapped because we want the original
        loss_real_adv_cls = F.cross_entropy(out_de_a, torch.ones(x_fake_a.size(0)))
        loss_real_adv_cls += F.cross_entropy(out_de_b, torch.zeros(x_fake_b.size(0)))
        
        if use_con_reg:
            out_de_aug_a = nets.discriminator.classifier(t(x_fake_a).detach())
            out_de_aug_b = nets.discriminator.classifier(t(x_fake_b).detach())
            loss_con_reg += F.smooth_l1_loss(out_de_a, out_de_aug_a)
            loss_con_reg += F.smooth_l1_loss(out_de_b, out_de_aug_b)
    else:
        loss_real_adv_cls = torch.zeros(1).mean()
        
    ## Second step adversarial loss - loss on cycle-consistent audios
    out_a = nets.discriminator(x_recon_a, torch.zeros(x_recon_a.size(0)))
    out_b = nets.discriminator(x_recon_b, torch.ones(x_recon_b.size(0)))
    loss_cycle = adv_loss(out_a, 0)
    loss_cycle += adv_loss(out_b, 0)
    if use_con_reg:
        out_aug_a = nets.discriminator(t(x_recon_a).detach(), torch.zeros(x_recon_a.size(0)))
        out_aug_b = nets.discriminator(t(x_recon_b).detach(), torch.ones(x_recon_b.size(0)))
        loss_con_reg += F.smooth_l1_loss(out_a, out_aug_a)
        loss_con_reg += F.smooth_l1_loss(out_b, out_aug_b)
    
    # adversarial classifier loss
    if use_adv_cls:
        out_de_a = nets.discriminator.classifier(x_recon_a)
        out_de_b = nets.discriminator.classifier(x_recon_b)
        ## here I let the ones and zeros as they are because I want the cycle-consistent audios
        # and basically the audio originates from the correct source sample
        loss_real_adv_cls += F.cross_entropy(out_de_a, torch.zeros(x_recon_a.size(0)))
        loss_real_adv_cls += F.cross_entropy(out_de_b, torch.ones(x_recon_b.size(0)))
        
        if use_con_reg:
            out_de_aug_a = nets.discriminator.classifier(t(x_recon_a).detach())
            out_de_aug_b = nets.discriminator.classifier(t(x_recon_b).detach())
            loss_con_reg += F.smooth_l1_loss(out_de_a, out_de_aug_a)
            loss_con_reg += F.smooth_l1_loss(out_de_b, out_de_aug_b)
    else:
        loss_real_adv_cls = torch.zeros(1).mean()
        
    ## Normalize the losses to the size of the input
    loss_real = loss_real * 0.5
    loss_fake = loss_fake * 0.5
    loss_cycle = loss_cycle * 0.5
    loss_reg = loss_reg * 0.25 # twice the speaker and also used again in cycle loss
    loss_real_adv_cls = loss_real_adv_cls * 0.25 # again twice the speaker and also used again in cycle loss
    
    loss = loss_real + loss_fake + loss_cycle + args.lambda_reg * loss_reg + \
            args.lambda_adv_cls * loss_real_adv_cls + \
            args.lambda_con_reg * loss_con_reg 
    ## because use two samples in each iteration

    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item(),
                       real_adv_cls=loss_real_adv_cls.item(),
                       con_reg=loss_con_reg.item())

def compute_g_loss(nets, args, x_real_a, x_real_b, use_adv_cls=False):
    args = Munch(args)
    # compute ASR/F0 features (real)
    with torch.no_grad():
        F0_real_a, GAN_F0_real_a, cyc_F0_real_a = nets.f0_model(x_real_a)
        F0_real_b, GAN_F0_real_b, cyc_F0_real_b = nets.f0_model(x_real_b)
        ASR_real_a = nets.asr_model.get_feature(x_real_a)
        ASR_real_b = nets.asr_model.get_feature(x_real_b)
    
    # adversarial loss
    x_fake_b = nets.generator_b(x_real_a, masks=None, F0=GAN_F0_real_a)
    x_identity_a = nets.generator_a(x_real_a, masks=None, F0=GAN_F0_real_a)
    x_recon_a = nets.generator_a(x_fake_b, masks=None, F0=GAN_F0_real_a)
    
    x_fake_a = nets.generator_a(x_real_b, masks=None, F0=GAN_F0_real_b)
    x_identity_b = nets.generator_b(x_real_b, masks=None, F0=GAN_F0_real_b)
    x_recon_b = nets.generator_b(x_fake_a, masks=None, F0=GAN_F0_real_b)
    
    
    
    out_a = nets.discriminator(x_fake_a, torch.zeros(x_fake_b.size(0)))
    out_b = nets.discriminator(x_fake_b, torch.ones(x_fake_a.size(0)))
    loss_adv = adv_loss(out_a, 1)
    loss_adv += adv_loss(out_b, 1)
    
    # compute ASR/F0 features (fake)
    F0_fake_a, GAN_F0_fake_a, _ = nets.f0_model(x_fake_a)
    F0_fake_b, GAN_F0_fake_b, _ = nets.f0_model(x_fake_b)
    ASR_fake_a = nets.asr_model.get_feature(x_fake_a)
    ASR_fake_b = nets.asr_model.get_feature(x_fake_b)
    
    # norm consistency loss
    x_fake_norm_a = log_norm(x_fake_a)
    x_real_norm_a = log_norm(x_real_a)
    x_fake_norm_b = log_norm(x_fake_b)
    x_real_norm_b = log_norm(x_real_b)
    loss_norm = ((torch.nn.ReLU()(torch.abs(x_fake_norm_a - x_real_norm_a) - args.norm_bias))**2).mean()
    loss_norm += ((torch.nn.ReLU()(torch.abs(x_fake_norm_b - x_real_norm_b) - args.norm_bias))**2).mean()
    
    # F0 loss
    loss_f0 = f0_loss(F0_fake_a, F0_real_a)
    loss_f0 += f0_loss(F0_fake_b, F0_real_b)
    
    # ASR loss
    loss_asr = F.smooth_l1_loss(ASR_fake_a, ASR_real_a)
    loss_asr += F.smooth_l1_loss(ASR_fake_b, ASR_real_b)
    
    # cycle-consistency loss
    loss_cyc = torch.mean(torch.abs(x_recon_a - x_real_a))
    loss_cyc += torch.mean(torch.abs(x_recon_b - x_real_b))
    
    # F0 loss in cycle-consistency loss
    if args.lambda_f0 > 0:
        _, _, cyc_F0_rec_a = nets.f0_model(x_recon_a)
        _, _, cyc_F0_rec_b = nets.f0_model(x_recon_b)
        loss_cyc += F.smooth_l1_loss(cyc_F0_rec_a, cyc_F0_real_a)
        loss_cyc += F.smooth_l1_loss(cyc_F0_rec_b, cyc_F0_real_b)
    if args.lambda_asr > 0:
        ASR_recon_a = nets.asr_model.get_feature(x_recon_a)
        ASR_recon_b = nets.asr_model.get_feature(x_recon_b)
        loss_cyc += F.smooth_l1_loss(ASR_recon_a, ASR_real_a)
        loss_cyc += F.smooth_l1_loss(ASR_recon_b, ASR_real_b)
    
    # adversarial classifier loss
    if use_adv_cls:
        out_de_a = nets.discriminator.classifier(x_fake_a)
        out_de_b = nets.discriminator.classifier(x_fake_b)
        loss_adv_cls = F.cross_entropy(out_de_a, torch.zeros(x_fake_a.size(0)))
        loss_adv_cls += F.cross_entropy(out_de_b, torch.ones(x_fake_b.size(0)))
    else:
        loss_adv_cls = torch.zeros(1).mean()
        
    ## normalize the losses to the number of speakers
    # mainly done for correct results in debugging
    loss_adv = loss_adv * 0.5
    loss_cyc = loss_cyc * 0.5
    loss_norm = loss_norm * 0.5
    loss_asr = loss_asr * 0.5
    loss_f0 = loss_f0 * 0.5
    loss_adv_cls = loss_adv_cls * 0.5
    
    loss = args.lambda_adv * loss_adv \
            + args.lambda_cyc * loss_cyc\
            + args.lambda_norm * loss_norm \
            + args.lambda_asr * loss_asr \
            + args.lambda_f0 * loss_f0 \
            + args.lambda_adv_cls * loss_adv_cls

    return loss, Munch(adv=loss_adv.item(),
                       cyc=loss_cyc.item(),
                       norm=loss_norm.item(),
                       asr=loss_asr.item(),
                       f0=loss_f0.item(),
                       adv_cls=loss_adv_cls.item())
    
# for norm consistency loss
def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x

# for adversarial loss
def adv_loss(logits, target):
    assert target in [1, 0]
    if len(logits.shape) > 1:
        logits = logits.reshape(-1)
    targets = torch.full_like(logits, fill_value=target)
    logits = logits.clamp(min=-10, max=10) # prevent nan
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss

# for R1 regularization loss
def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg

# for F0 consistency loss
def compute_mean_f0(f0):
    f0_mean = f0.mean(-1)
    f0_mean = f0_mean.expand(f0.shape[-1], f0_mean.shape[0]).transpose(0, 1) # (B, M)
    return f0_mean

def f0_loss(x_f0, y_f0):
    """
    x.shape = (B, 1, M, L): predict
    y.shape = (B, 1, M, L): target
    """
    # compute the mean
    x_mean = compute_mean_f0(x_f0)
    y_mean = compute_mean_f0(y_f0)
    loss = F.l1_loss(x_f0 / x_mean, y_f0 / y_mean)
    return loss