import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from models import losses
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
import pointnet2_ops.pointnet2_modules as pointnet2
import numpy as np
import utils.utils as utils
from utils.visualization_utils import *

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':

        def lambda_rule(epoch):
            lr_l = 1.0 - max(
                0, epoch + 1 + 1 - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=opt.lr_decay_iters,
                                        gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.2,
                                                   threshold=0.01,
                                                   patience=5)
    else:
        return NotImplementedError(
            'learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type, init_gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' %
                    init_type)
        elif classname.find('BatchNorm') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def init_net(net, init_type, init_gain, gpu_ids):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        net = net.cuda()
        net = torch.nn.DataParallel(net, gpu_ids)
    if init_type != 'none':
        init_weights(net, init_type, init_gain)
    return net


def define_classifier(opt, gpu_ids, arch, init_type, init_gain, device):
    net = None
    if arch == 'vae':
        net = GraspSamplerVAE(opt.model_scale, opt.pointnet_radius,
                              opt.pointnet_nclusters, opt.latent_size, device, opt.dual_grasp, opt.merge_pcs_in_vae_encoder)
    elif arch == 'gan':
        net = GraspSamplerGAN(opt.model_scale, opt.pointnet_radius,
                              opt.pointnet_nclusters, opt.latent_size, device, opt.dual_grasp)
    elif arch == 'evaluator':
        net = GraspEvaluator(opt.model_scale, opt.pointnet_radius,
                             opt.pointnet_nclusters, device)
    else:
        raise NotImplementedError('model name [%s] is not recognized' % arch)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_loss(opt):
    if opt.arch == 'vae':
        if opt.split_decoders:
            kl_loss = losses.kl_divergence_split
            # reconstruction_loss = losses.control_point_l1_loss_split
            reconstruction_loss = losses.min_distance_loss_split
            print("Using split decoders")
        else:
            kl_loss = losses.kl_divergence
            # reconstruction_loss = losses.control_point_l1_loss
            reconstruction_loss = losses.min_distance_loss
        return kl_loss, reconstruction_loss
    elif opt.arch == 'gan':
        if opt.split_decoders:
            reconstruction_loss = losses.min_distance_loss_split
        else:
            reconstruction_loss = losses.min_distance_loss
        return reconstruction_loss
    elif opt.arch == 'evaluator':
        loss = losses.classification_with_confidence_loss
        return loss
    else:
        raise NotImplementedError("Loss not found")


class GraspSampler(nn.Module):
    def __init__(self, latent_size, device):
        super(GraspSampler, self).__init__()
        self.latent_size = latent_size
        self.device = device

    def create_decoder(self, model_scale, pointnet_radius, pointnet_nclusters,
                       num_input_features, dual_grasp=False):
        # The number of input features for the decoder is 3+latent space where 3
        # represents the x, y, z position of the point-cloud        
        
        if dual_grasp:
            self.decoder_left = base_network(pointnet_radius, pointnet_nclusters,
                                    model_scale, num_input_features)
            self.decoder_right = base_network(pointnet_radius, pointnet_nclusters,
                                    model_scale, num_input_features)
            self.dual_grasp = True
            self.qleft = nn.Linear(model_scale * 1024, 4)
            self.tleft = nn.Linear(model_scale * 1024, 3)
            self.qright = nn.Linear(model_scale * 1024, 4)
            self.tright = nn.Linear(model_scale * 1024, 3)
            self.confidence_left = nn.Linear(model_scale * 1024, 1)
            self.confidence_right = nn.Linear(model_scale * 1024, 1)
        else:
            self.decoder = base_network(pointnet_radius, pointnet_nclusters,
                                    model_scale, num_input_features)
            self.dual_grasp = False
            self.q = nn.Linear(model_scale * 1024, 4)
            self.t = nn.Linear(model_scale * 1024, 3)
            self.confidence = nn.Linear(model_scale * 1024, 1)

    def decode(self, xyz, z, dual_grasp=False):
        
        dual_grasp = self.dual_grasp
        if dual_grasp:
            z_left = z[0]
            z_right = z[1]
            xyz_left = xyz.clone()
            xyz_right = xyz.clone()
            xyz_features_left = self.concatenate_z_with_pc(xyz,
                                                    z_left).transpose(-1,
                                                                1).contiguous()
            xyz_features_right = self.concatenate_z_with_pc(xyz,
                                                    z_right).transpose(-1,
                                                                1).contiguous()
            for module in self.decoder_left[0]:
                xyz_left, xyz_features_left = module(xyz_left, xyz_features_left)
            x_left = self.decoder_left[1](xyz_features_left.squeeze(-1))
            for module in self.decoder_right[0]:
                xyz_right, xyz_features_right = module(xyz_right, xyz_features_right)
            x_right = self.decoder_right[1](xyz_features_right.squeeze(-1))
            # predicted_qt = torch.cat(
            #     (F.normalize(self.qleft(x), p=2, dim=-1), self.tleft(x),
            #     F.normalize(self.qright(x), p=2, dim=-1), self.tright(x)) -1)
            predicted_qtleft = torch.cat(
                (F.normalize(self.qleft(x_left), p=2, dim=-1), self.tleft(x_left)), -1)
            predicted_qtright = torch.cat(
                (F.normalize(self.qright(x_right), p=2, dim=-1), self.tright(x_right)), -1)
            # print(predicted_qtleft.shape, predicted_qtright.shape)
            #Combine the two grasps with size (batch_size, 14)
            # predicted_qt = torch.cat((predicted_qtleft, predicted_qtright), -1)
            # print(predicted_qt.shape)
            confidence_left = torch.sigmoid(self.confidence_left(x_left)).squeeze()
            confidence_right = torch.sigmoid(self.confidence_right(x_right)).squeeze()
            # print(confidence_left.shape, confidence_right.shape)
            # confidence = torch.stack((confidence_left, confidence_right), dim = 1)
            # print(confidence.shape)
            return (predicted_qtleft, predicted_qtright), (confidence_left, confidence_right)
        else:
            xyz_features = self.concatenate_z_with_pc(xyz,
                                                  z).transpose(-1,
                                                               1).contiguous()
            for module in self.decoder[0]:
                xyz, xyz_features = module(xyz, xyz_features)
            x = self.decoder[1](xyz_features.squeeze(-1))
            predicted_qt = torch.cat(
                (F.normalize(self.q(x), p=2, dim=-1), self.t(x)), -1)
        # print(predicted_qt.shape)
            return predicted_qt, torch.sigmoid(self.confidence(x)).squeeze()

    def concatenate_z_with_pc(self, pc, z):
        z.unsqueeze_(1)
        z = z.expand(-1, pc.shape[1], -1)
        return torch.cat((pc, z), -1)

    def get_latent_size(self):
        return self.latent_size


class GraspSamplerVAE(GraspSampler):
    """Network for learning a generative VAE grasp-sampler
    """
    def __init__(self,
                 model_scale,
                 pointnet_radius=0.02,
                 pointnet_nclusters=128,
                 latent_size=2,
                 device="cpu",
                 dual_grasp=False,
                 merge_pcs_in_vae_encoder=False):
        super(GraspSamplerVAE, self).__init__(latent_size, device)
        self.create_encoder(model_scale, pointnet_radius, pointnet_nclusters, dual_grasp, merge_pcs_in_vae_encoder)

        self.create_decoder(model_scale, pointnet_radius, pointnet_nclusters,
                            latent_size + 3, dual_grasp)
        self.create_bottleneck(model_scale * 1024, latent_size)
        self.merge_pcs_in_vae_encoder = merge_pcs_in_vae_encoder

    def create_encoder(
            self,
            model_scale,
            pointnet_radius,
            pointnet_nclusters,
            dual_grasp=False,
            merge_pcs_in_vae_encoder=False
    ):
        # The number of input features for the encoder is 19: the x, y, z
        # position of the point-cloud and the flattened 4x4=16 grasp pose matrix
        #If using dual grasps the number of the input features is 35: the x, y, z
        # position of the point-cloud and the flattened 2x4x4=32 grasp pose matrix
        # print("dual_grasp", dual_grasp)
        if dual_grasp and not merge_pcs_in_vae_encoder:
            self.encoder_left = base_network(pointnet_radius, pointnet_nclusters,
                                        model_scale, 19)
            self.encoder_right = base_network(pointnet_radius, pointnet_nclusters,
                                        model_scale, 19)
            # self.encoder_left = base_network(pointnet_radius, pointnet_nclusters,
            #                             model_scale, 19)
            # self.encoder_right = base_network(pointnet_radius, pointnet_nclusters,
            #                             model_scale, 19)
        elif dual_grasp and merge_pcs_in_vae_encoder:
            self.encoder_left = base_network(pointnet_radius, pointnet_nclusters,
                                        model_scale, 4)
            self.encoder_right = base_network(pointnet_radius, pointnet_nclusters,
                                        model_scale, 4)
        else:
            self.encoder = base_network(pointnet_radius, pointnet_nclusters,
                                    model_scale, 19)

    def create_bottleneck(self, input_size, latent_size):
        mu = nn.Linear(input_size, latent_size)
        logvar = nn.Linear(input_size, latent_size)
        self.latent_space = nn.ModuleList([mu, logvar])

    def encode(self, xyz, xyz_features, encoder = None):
        # print(xyz.shape, xyz_features.shape)
       
        if self.dual_grasp and not self.merge_pcs_in_vae_encoder: 
            xyz_left = xyz
            xyz_right = xyz
            xyz_features_left = xyz_features[:, 0, :, :]
            xyz_features_right = xyz_features[:, 1, :, :]
            # print(xyz_features_left.shape, xyz_features_right.shape)
            for module in self.encoder_left[0]:
                xyz_left, xyz_features_left = module(xyz_left, xyz_features_left.contiguous())
            for module in self.encoder_right[0]:
                xyz_right, xyz_features_right = module(xyz_right, xyz_features_right.contiguous())
            # xyz_features = torch.cat((xyz_features_left, xyz_features_right), -1)
            return (self.encoder_left[1](xyz_features_left.squeeze(-1)), self.encoder_right[1](xyz_features_right.squeeze(-1)))
        elif self.dual_grasp and self.merge_pcs_in_vae_encoder:
            if encoder == "left":
                for module in self.encoder_left[0]:
                    xyz, xyz_features = module(xyz, xyz_features.contiguous())
                return self.encoder_left[1](xyz_features.squeeze(-1))
            elif encoder == "right":
                for module in self.encoder_right[0]:
                    xyz, xyz_features = module(xyz, xyz_features.contiguous())
                return self.encoder_right[1](xyz_features.squeeze(-1))
        else:
            for module in self.encoder[0]:
                xyz, xyz_features = module(xyz, xyz_features)
            return self.encoder[1](xyz_features.squeeze(-1))

    def bottleneck(self, z):
        return self.latent_space[0](z), self.latent_space[1](z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, pc, grasp=None, train=True):
        if train and not self.merge_pcs_in_vae_encoder:
            # print(grasp.shape)
            return self.forward_train(pc, grasp)
        elif train and self.merge_pcs_in_vae_encoder:
            return self.forward_train_merge(pc, grasp)
        elif not train and not self.merge_pcs_in_vae_encoder:
            return self.forward_test(pc, grasp)
        elif not train and self.merge_pcs_in_vae_encoder:
            return self.forward_test_merge(pc, grasp)
    
    def forward_train(self, pc, grasp):
        # print(pc.shape, grasp.shape)
        if self.dual_grasp:
            grasp = grasp.reshape(len(grasp[0]),2, 16)
            left_grasp = grasp[:, 0, :]
            right_grasp = grasp[:, 1, :]
            input_features_left = torch.cat(
                (pc, left_grasp.unsqueeze(1).expand(-1, pc.shape[1], -1)),
                -1).transpose(-1, 1).contiguous()
            input_features_right = torch.cat(
                (pc, right_grasp.unsqueeze(1).expand(-1, pc.shape[1], -1)),
                -1).transpose(-1, 1).contiguous()
            #reshape the input features to batch x 2 x 19 x 1024
            input_features = torch.stack((input_features_left, input_features_right), dim = 1)
            z = self.encode(pc, input_features)
            mu_left, logvar_left = self.bottleneck(z[0])
            mu_right, logvar_right = self.bottleneck(z[1])
            z_left = self.reparameterize(mu_left, logvar_left)
            z_right = self.reparameterize(mu_right, logvar_right)
            z = (z_left, z_right)
            qt, confidence = self.decode(pc, z, self.dual_grasp)
            # print("mu shapes", mu_left.shape, mu_right.shape)
            # mu = torch.cat((mu_left, mu_right), -1)
            mu = (mu_left, mu_right)
            # print("mu shape", mu.shape) 
            # print(logvar_left.shape, logvar_right.shape)\
            # logvar = torch.cat((logvar_left, logvar_right), -1)
            logvar = (logvar_left, logvar_right)
        else:
            input_features = torch.cat(
                (pc, grasp.unsqueeze(1).expand(-1, pc.shape[1], -1)),
                -1).transpose(-1, 1).contiguous()
            z = self.encode(pc, input_features)
            mu, logvar = self.bottleneck(z)
            print("mu shape", mu.shape)
            print("logvar shape", logvar.shape)
            z = self.reparameterize(mu, logvar)
            qt, confidence = self.decode(pc, z, self.dual_grasp)
        # print(mu.shape, logvar.shape)
        ###CODE SNIPPET TO VISUALIZE THE GENERATED CONTROL POINTS
        # predicted_cp = utils.transform_control_points(
        #         qt, qt.shape[0], device=self.device, dual_grasp = self.dual_grasp)
        # if len(predicted_cp.shape) == 4:
        #     predicted_cp = predicted_cp.reshape(-1, 6, 3)

        # mlab.figure(bgcolor=(1, 1, 1))
        # draw_scene(
        #         pc[0].cpu().detach().numpy(),
        #         # grasps=self.og_grasps,
        #         target_cps=predicted_cp,
        #     )
        # mlab.show()
        # print(xd)
        ###END OF CODE SNIPPET
        return qt, confidence, mu, logvar

    def forward_train_merge(self, pc, gripper_pc):
        if self.dual_grasp:
            gripper_pc_clone = gripper_pc.clone()
            gripper_pc_left = gripper_pc[:, 0, :, :]
            gripper_pc_right = gripper_pc_clone[:, 1, :, :]
            pc_left = pc.clone()
            pc_right = pc.clone()
            pc_left, pc_features_left = self.merge_pc_and_gripper_pc(pc_left, gripper_pc_left)
            pc_right, pc_features_right = self.merge_pc_and_gripper_pc(pc_right, gripper_pc_right)
            z_left = self.encode(pc_left, pc_features_left.contiguous(), encoder = "left")
            z_right = self.encode(pc_right, pc_features_right.contiguous(), encoder = "right")
            mu_left, logvar_left = self.bottleneck(z_left)
            mu_right, logvar_right = self.bottleneck(z_right)
            z_left = self.reparameterize(mu_left, logvar_left)
            z_right = self.reparameterize(mu_right, logvar_right)
            z = (z_left, z_right)
            mu = (mu_left, mu_right)
            logvar = (logvar_left, logvar_right)
            qt, confidence = self.decode(pc, z, self.dual_grasp)
        else:
            pc, pc_features = self.merge_pc_and_gripper_pc(pc, gripper_pc)
            z = self.encode(pc, pc_features.contiguous())
            mu, logvar = self.bottleneck(z)
            z = self.reparameterize(mu, logvar)
            qt, confidence = self.decode(pc, z, self.dual_grasp)
        return qt, confidence, mu, logvar

    def forward_test(self, pc, grasp):
        if self.dual_grasp:
            grasp = grasp.reshape(len(grasp[0]),2, 16)
            left_grasp = grasp[:, 0, :]
            right_grasp = grasp[:, 1, :]
            input_features_left = torch.cat(
                (pc, left_grasp.unsqueeze(1).expand(-1, pc.shape[1], -1)),
                -1).transpose(-1, 1).contiguous()
            input_features_right = torch.cat(
                (pc, right_grasp.unsqueeze(1).expand(-1, pc.shape[1], -1)),
                -1).transpose(-1, 1).contiguous()
            #reshape the input features to batch x 2 x 19 x 1024
            input_features = torch.stack((input_features_left, input_features_right), dim = 1)
            z = self.encode(pc, input_features)
            mu_left, logvar_left = self.bottleneck(z[0])
            mu_right, logvar_right = self.bottleneck(z[1])
            z_left = self.reparameterize(mu_left, logvar_left)
            z_right = self.reparameterize(mu_right, logvar_right)
            z = (z_left, z_right)
            # mu = torch.cat((mu_left, mu_right), -1)
            mu = (mu_left, mu_right)
            qt, confidence = self.decode(pc, mu, self.dual_grasp)
        else:
            input_features = torch.cat(
                (pc, grasp.unsqueeze(1).expand(-1, pc.shape[1], -1)),
                -1).transpose(-1, 1).contiguous()
            z = self.encode(pc, input_features)
            mu, _ = self.bottleneck(z)
            qt, confidence = self.decode(pc, mu, self.dual_grasp)
        return qt, confidence
    
    def forward_test_merge(self, pc, gripper_pc):
        if self.dual_grasp:
            gripper_pc_left = gripper_pc[:, 0, :, :]
            gripper_pc_right = gripper_pc[:, 1, :, :]
            pc_left = pc.clone()
            pc_right = pc.clone()
            pc_left, pc_features_left = self.merge_pc_and_gripper_pc(pc_left, gripper_pc_left)
            pc_right, pc_features_right = self.merge_pc_and_gripper_pc(pc_right, gripper_pc_right)
            z_left = self.encode(pc_left, pc_features_left.contiguous(), encoder = "left")
            z_right = self.encode(pc_right, pc_features_right.contiguous(), encoder = "right")
            z = (z_left, z_right)
            mu_left, _ = self.bottleneck(z_left)
            mu_right, _ = self.bottleneck(z_right)
            mu = (mu_left, mu_right)
            qt, confidence = self.decode(pc, mu, self.dual_grasp)
        else:
            pc, pc_features = self.merge_pc_and_gripper_pc(pc, gripper_pc)
            z = self.encode(pc, pc_features.contiguous())
            mu, _ = self.bottleneck(z)
            qt, confidence = self.decode(pc, mu, self.dual_grasp)
        return qt, confidence

    def sample_latent(self, batch_size):
        return torch.randn(batch_size, self.latent_size).to(self.device)

    def generate_grasps(self, pc, z=None):
        if z is None:
            z_left = self.sample_latent(pc.shape[0])
            z_right = self.sample_latent(pc.shape[0])
            z = (z_left, z_right)
        qt, confidence = self.decode(pc, z, self.dual_grasp)
        predicted_cp = qt
        predicted_cp_left = utils.transform_control_points(
                    predicted_cp[0], predicted_cp[0].shape[0], device=self.device, dual_grasp = False)
        predicted_cp_right = utils.transform_control_points(
            predicted_cp[1], predicted_cp[1].shape[0], device=self.device, dual_grasp = False)
        predicted_cp = [predicted_cp_left, predicted_cp_right]
        if isinstance(qt, tuple):
            qt = torch.cat((qt[0], qt[1]), -1)
            confidence = torch.stack((confidence[0], confidence[1]), 1)
            z = torch.cat((z[0], z[1]), -1)
        # if len(predicted_cp.shape) == 4:
        #     predicted_cp = predicted_cp.reshape(-1, 6, 3)

        # mlab.figure(bgcolor=(1, 1, 1))
        # draw_scene(
        #         pc[0].cpu().detach().numpy(),
        #         # grasps=self.og_grasps,
        #         target_cps=predicted_cp,
        #     )
        # mlab.show()
        # print(xd)
        
        return qt, confidence, z.squeeze()

    def generate_dense_latents(self, resolution):
        """
        For the VAE sampler we consider dense latents to correspond to those between -2 and 2
        """
        latents = torch.meshgrid(*[
            torch.linspace(-2, 2, resolution) for i in range(self.latent_size)
        ])
        return torch.stack([latents[i].flatten() for i in range(len(latents))],
                           dim=-1).to(self.device)
    def merge_pc_and_gripper_pc(self, pc, gripper_pc):
        """
        Merges the object point cloud and gripper point cloud and
        adds a binary auxiliary feature that indicates whether each point
        belongs to the object or to the gripper.
        """
        # print(gripper_pc.shape)
        pc_shape = pc.shape
        gripper_shape = gripper_pc.shape
        if len(gripper_pc.shape) == 4:
            #Code snippet that creates seperate combined point clouds for each gripper
            #make two gripper point clouds of size (batch_size, 6, 3)
            # pc1 = pc.clone()
            # pc2 = pc.clone()
            # gripper_pc1 = gripper_pc[:, 0, :, :]
            # gripper_pc2 = gripper_pc[:, 1, :, :]
            # print(gripper_pc1.shape, gripper_pc2.shape)
            # l0_xyz1 = torch.cat((pc1, gripper_pc1), 1)
            # print(l0_xyz1.shape)
            # l0_xyz2 = torch.cat((pc2, gripper_pc2), 1)
            # labels1 = [
            #     torch.ones(pc.shape[1], 1, dtype=torch.float32),
            #     torch.zeros(gripper_pc1.shape[1], 1, dtype=torch.float32)
            # ]
            # labels1 = torch.cat(labels1, 0)
            # labels1.unsqueeze_(0)
            # labels1 = labels1.repeat(pc_shape[0], 1, 1)
            # labels2 = [
            #     torch.ones(pc.shape[1], 1, dtype=torch.float32),
            #     torch.zeros(gripper_pc2.shape[1], 1, dtype=torch.float32)
            # ]
            # labels2 = torch.cat(labels2, 0)
            # labels2.unsqueeze_(0)
            # labels2 = labels2.repeat(pc_shape[0], 1, 1)
            # l0_points1 = torch.cat([l0_xyz1, labels1.to(self.device)],
            #                        -1).transpose(-1, 1)
            # l0_points2 = torch.cat([l0_xyz2, labels2.to(self.device)],
            #                           -1).transpose(-1, 1)
            # print( torch.cat((l0_xyz1, l0_xyz2), 0).shape, torch.cat((l0_points1, l0_points2), 0).shape)
            # return torch.cat((l0_xyz1, l0_xyz2), 0), torch.cat((l0_points1, l0_points2), 0)
            #End of code snippet
            #Code snippet that creates one point cloud with both grippers
            gripper_pc1 = gripper_pc[:, 0, :, :]
            gripper_pc2 = gripper_pc[:, 1, :, :]
            l0_xyz = torch.cat((pc, gripper_pc1, gripper_pc2), 1)
            # print(l0_xyz.shape)
            labels = [
                torch.ones(pc.shape[1], 1, dtype=torch.float32),
                torch.zeros(gripper_pc1.shape[1], 1, dtype=torch.float32),
                torch.zeros(gripper_pc2.shape[1], 1, dtype=torch.float32)
            ]
            labels = torch.cat(labels, 0)
            labels.unsqueeze_(0)
            labels = labels.repeat(pc_shape[0], 1, 1)
            l0_points = torch.cat([l0_xyz, labels.to(self.device)],
                                  -1).transpose(-1, 1)
            return l0_xyz, l0_points


        

        assert (len(pc_shape) == 3)
        assert (len(gripper_shape) == 3)
        assert (pc_shape[0] == gripper_shape[0])

        npoints = pc_shape[1]
        batch_size = pc_shape[0]

        l0_xyz = torch.cat((pc, gripper_pc), 1)
        labels = [
            torch.ones(pc.shape[1], 1, dtype=torch.float32),
            torch.zeros(gripper_pc.shape[1], 1, dtype=torch.float32)
        ]
        labels = torch.cat(labels, 0)
        labels.unsqueeze_(0)
        labels = labels.repeat(batch_size, 1, 1)

        l0_points = torch.cat([l0_xyz, labels.to(self.device)],
                              -1).transpose(-1, 1)
        return l0_xyz, l0_points


class GraspSamplerGAN(GraspSampler):
    """
    Altough the name says this sampler is based on the GAN formulation, it is
    not actually optimizing based on the commonly known adversarial game.
    Instead, it is based on the Implicit Maximum Likelihood Estimation from
    https://arxiv.org/pdf/1809.09087.pdf which is similar to the GAN formulation
    but with new insights that avoids e.g. mode collapses.
    """
    def __init__(self,
                 model_scale,
                 pointnet_radius,
                 pointnet_nclusters,
                 latent_size=2,
                 device="cpu",
                 dual_grasp=False):
        super(GraspSamplerGAN, self).__init__(latent_size, device)
        self.create_decoder(model_scale, pointnet_radius, pointnet_nclusters,
                            latent_size + 3, dual_grasp=dual_grasp)

    def sample_latent(self, batch_size):
        return torch.rand(batch_size, self.latent_size).to(self.device)

    def forward(self, pc, grasps=None, train=True):

        z_left = self.sample_latent(pc.shape[0])
        z_right = self.sample_latent(pc.shape[0])
        z = (z_left, z_right)
        return self.decode(pc, z)

    def generate_grasps(self, pc, z=None):
        if z is None:
            z_left = self.sample_latent(pc.shape[0])
            z_right = self.sample_latent(pc.shape[0])
            z = (z_left, z_right)
        qt, confidence = self.decode(pc, z)
        
        # predicted_cp = utils.transform_control_points(
        #         qt, qt.shape[0], device=self.device, dual_grasp = self.dual_grasp)
        # if len(predicted_cp.shape) == 4:
        #     predicted_cp = predicted_cp.reshape(-1, 6, 3)

        # mlab.figure(bgcolor=(1, 1, 1))
        # draw_scene(
        #         pc[0].cpu().detach().numpy(),
        #         # grasps=self.og_grasps,
        #         target_cps=predicted_cp,
        #     )
        # mlab.show()
        # print(xd)
        #Check if the qt, confidence and z are tuples
        if isinstance(qt, tuple):
            qt = torch.cat((qt[0], qt[1]), -1)
            confidence = torch.stack((confidence[0], confidence[1]), 1)
            z = torch.cat((z[0], z[1]), -1)
        # print("qt", qt.shape, "confidence", confidence.shape)
        return qt, confidence, z.squeeze()

    def generate_dense_latents(self, resolution):
        latents = torch.meshgrid(*[
            torch.linspace(0, 1, resolution) for i in range(self.latent_size)
        ])
        return torch.stack([latents[i].flatten() for i in range(len(latents))],
                           dim=-1).to(self.device)


class GraspEvaluator(nn.Module):
    def __init__(self,
                 model_scale=1,
                 pointnet_radius=0.02,
                 pointnet_nclusters=128,
                 device="cpu"):
        super(GraspEvaluator, self).__init__()
        self.create_evaluator(pointnet_radius, model_scale, pointnet_nclusters)
        self.device = device

    def create_evaluator(self, pointnet_radius, model_scale,
                         pointnet_nclusters):
        # The number of input features for the evaluator is 4: the x, y, z
        # position of the concatenated gripper and object point-clouds and an
        # extra binary feature, which is 0 for the object and 1 for the gripper,
        # to tell these point-clouds apart
        self.evaluator = base_network(pointnet_radius, pointnet_nclusters,
                                      model_scale, 4)
        self.predictions_logits = nn.Linear(1024 * model_scale, 1)
        self.confidence = nn.Linear(1024 * model_scale, 1)

    def evaluate(self, xyz, xyz_features):
        # print("test",xyz.shape, xyz_features.shape)
        for module in self.evaluator[0]:
            xyz, xyz_features = module(xyz, xyz_features)
        # print(self.evaluator[1](xyz_features.squeeze(-1)).shape)
        return self.evaluator[1](xyz_features.squeeze(-1))

    def forward(self, pc, gripper_pc, train=True):
        pc, pc_features = self.merge_pc_and_gripper_pc(pc, gripper_pc)
        x = self.evaluate(pc, pc_features.contiguous())
        return self.predictions_logits(x), torch.sigmoid(self.confidence(x))

    def merge_pc_and_gripper_pc(self, pc, gripper_pc):
        """
        Merges the object point cloud and gripper point cloud and
        adds a binary auxiliary feature that indicates whether each point
        belongs to the object or to the gripper.
        """
        # print(gripper_pc.shape)
        pc_shape = pc.shape
        gripper_shape = gripper_pc.shape
        if len(gripper_pc.shape) == 4:
            #Code snippet that creates seperate combined point clouds for each gripper
            #make two gripper point clouds of size (batch_size, 6, 3)
            # pc1 = pc.clone()
            # pc2 = pc.clone()
            # gripper_pc1 = gripper_pc[:, 0, :, :]
            # gripper_pc2 = gripper_pc[:, 1, :, :]
            # print(gripper_pc1.shape, gripper_pc2.shape)
            # l0_xyz1 = torch.cat((pc1, gripper_pc1), 1)
            # print(l0_xyz1.shape)
            # l0_xyz2 = torch.cat((pc2, gripper_pc2), 1)
            # labels1 = [
            #     torch.ones(pc.shape[1], 1, dtype=torch.float32),
            #     torch.zeros(gripper_pc1.shape[1], 1, dtype=torch.float32)
            # ]
            # labels1 = torch.cat(labels1, 0)
            # labels1.unsqueeze_(0)
            # labels1 = labels1.repeat(pc_shape[0], 1, 1)
            # labels2 = [
            #     torch.ones(pc.shape[1], 1, dtype=torch.float32),
            #     torch.zeros(gripper_pc2.shape[1], 1, dtype=torch.float32)
            # ]
            # labels2 = torch.cat(labels2, 0)
            # labels2.unsqueeze_(0)
            # labels2 = labels2.repeat(pc_shape[0], 1, 1)
            # l0_points1 = torch.cat([l0_xyz1, labels1.to(self.device)],
            #                        -1).transpose(-1, 1)
            # l0_points2 = torch.cat([l0_xyz2, labels2.to(self.device)],
            #                           -1).transpose(-1, 1)
            # print( torch.cat((l0_xyz1, l0_xyz2), 0).shape, torch.cat((l0_points1, l0_points2), 0).shape)
            # return torch.cat((l0_xyz1, l0_xyz2), 0), torch.cat((l0_points1, l0_points2), 0)
            #End of code snippet
            #Code snippet that creates one point cloud with both grippers
            gripper_pc1 = gripper_pc[:, 0, :, :]
            gripper_pc2 = gripper_pc[:, 1, :, :]
            l0_xyz = torch.cat((pc, gripper_pc1, gripper_pc2), 1)
            # print(l0_xyz.shape)
            labels = [
                torch.ones(pc.shape[1], 1, dtype=torch.float32),
                torch.zeros(gripper_pc1.shape[1], 1, dtype=torch.float32),
                torch.zeros(gripper_pc2.shape[1], 1, dtype=torch.float32)
            ]
            labels = torch.cat(labels, 0)
            labels.unsqueeze_(0)
            labels = labels.repeat(pc_shape[0], 1, 1)
            l0_points = torch.cat([l0_xyz, labels.to(self.device)],
                                  -1).transpose(-1, 1)
            return l0_xyz, l0_points


        

        assert (len(pc_shape) == 3)
        assert (len(gripper_shape) == 3)
        assert (pc_shape[0] == gripper_shape[0])

        npoints = pc_shape[1]
        batch_size = pc_shape[0]

        l0_xyz = torch.cat((pc, gripper_pc), 1)
        labels = [
            torch.ones(pc.shape[1], 1, dtype=torch.float32),
            torch.zeros(gripper_pc.shape[1], 1, dtype=torch.float32)
        ]
        labels = torch.cat(labels, 0)
        labels.unsqueeze_(0)
        labels = labels.repeat(batch_size, 1, 1)

        l0_points = torch.cat([l0_xyz, labels.to(self.device)],
                              -1).transpose(-1, 1)
        return l0_xyz, l0_points


def base_network(pointnet_radius, pointnet_nclusters, scale, in_features):
    sa1_module = pointnet2.PointnetSAModule(
        npoint=pointnet_nclusters,
        radius=pointnet_radius,
        nsample=64,
        mlp=[in_features, 64 * scale, 64 * scale, 128 * scale])
    sa2_module = pointnet2.PointnetSAModule(
        npoint=32,
        radius=0.04,
        nsample=128,
        mlp=[128 * scale, 128 * scale, 128 * scale, 256 * scale])

    sa3_module = pointnet2.PointnetSAModule(
        mlp=[256 * scale, 256 * scale, 256 * scale, 512 * scale])

    sa_modules = nn.ModuleList([sa1_module, sa2_module, sa3_module])
    fc_layer = nn.Sequential(nn.Linear(512 * scale, 1024 * scale),
                             nn.BatchNorm1d(1024 * scale), nn.ReLU(True),
                             nn.Linear(1024 * scale, 1024 * scale),
                             nn.BatchNorm1d(1024 * scale), nn.ReLU(True))
    return nn.ModuleList([sa_modules, fc_layer])

