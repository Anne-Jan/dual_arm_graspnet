import torch
from . import networks
from os.path import join
import utils.utils as utils
from utils.visualization_utils import *


class GraspNetModel:
    """ Class for training Model weights

    :args opt: structure containing configuration params
    e.g.,
    --dataset_mode -> sampling / evaluation)
    """
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        if self.gpu_ids and self.gpu_ids[0] >= torch.cuda.device_count():
            self.gpu_ids[0] = torch.cuda.device_count() - 1
        self.device = torch.device('cuda:{}'.format(
            self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = join(opt.checkpoints_dir, opt.name)
        self.optimizer = None
        self.loss = None
        self.pcs = None
        self.grasps = None
        self.dual_grasp = opt.dual_grasp
        # print(opt)
        self.split_decoders = opt.split_decoders
        print("split_decoders", self.split_decoders)
        # load/define networks
        self.net = networks.define_classifier(opt, self.gpu_ids, opt.arch,
                                              opt.init_type, opt.init_gain,
                                              self.device)

        self.criterion = networks.define_loss(opt)

        self.confidence_loss = None
        if self.opt.arch == "vae":
            self.kl_loss = None
            self.reconstruction_loss = None
        elif self.opt.arch == "gan":
            self.reconstruction_loss = None
        else:
            self.classification_loss = None

        if self.is_train and opt.split_decoders:
            if self.opt.arch == "vae":
                self.optimizer_left = torch.optim.Adam(list(self.net.module.encoder_left.parameters()) + list(self.net.module.decoder_left.parameters()) + list(self.net.module.qleft.parameters()) + list(self.net.module.tleft.parameters()) + list(self.net.module.confidence_left.parameters()),
                                                lr=opt.lr,
                                                betas=(opt.beta1, 0.999))
                self.optimizer_right = torch.optim.Adam(list(self.net.module.encoder_right.parameters()) + list(self.net.module.decoder_right.parameters()) + list(self.net.module.qright.parameters()) + list(self.net.module.tright.parameters()) + list(self.net.module.confidence_right.parameters()),
                                                lr=opt.lr,
                                                betas=(opt.beta1, 0.999))
                self.scheduler_left = networks.get_scheduler(self.optimizer_left, opt)
                self.scheduler_right = networks.get_scheduler(self.optimizer_right, opt)
            if self.opt.arch == "gan":
                self.optimizer_left = torch.optim.Adam(list(self.net.module.decoder_left.parameters()) + list(self.net.module.qleft.parameters()) + list(self.net.module.tleft.parameters()) + list(self.net.module.confidence_left.parameters()),
                                                lr=opt.lr,
                                                betas=(opt.beta1, 0.999))
                self.optimizer_right = torch.optim.Adam(list(self.net.module.decoder_right.parameters()) + list(self.net.module.qright.parameters()) + list(self.net.module.tright.parameters()) + list(self.net.module.confidence_right.parameters()),
                                                lr=opt.lr,
                                                betas=(opt.beta1, 0.999))
                self.scheduler_left = networks.get_scheduler(self.optimizer_left, opt)
                self.scheduler_right = networks.get_scheduler(self.optimizer_right, opt)
        elif self.is_train and not opt.split_decoders:
            self.optimizer = torch.optim.Adam(self.net.parameters(),
                                              lr=opt.lr,
                                              betas=(opt.beta1, 0.999))
            self.scheduler = networks.get_scheduler(self.optimizer, opt)
        if not self.is_train or opt.continue_train:
            self.load_network(opt.which_epoch, self.is_train)

    def set_input(self, data):
        input_pcs = torch.from_numpy(data['pc']).contiguous()
        self.pc_for_visualization = input_pcs
        if self.opt.merge_pcs_in_vae_encoder:
            input_grasps= torch.from_numpy(data['target_cps']).float()
        else:
            input_grasps = torch.from_numpy(data['grasp_rt']).float()
        #check if the input grasps and pc are not nan
        if torch.isnan(input_pcs).any() or torch.isnan(input_grasps).any():
            print("Input pc or grasps are nan")
        # self.og_grasps = data['og_grasps']
        # # print("input_grasps", input_grasps.shape)
        # # print(len(self.og_grasps))
        # # reshape from 32 x 2 x 4 x 4 to 64 x 4 x 4
        # # ##CODE SNIPPET TO VISUALIZE THE INPUT GRASPS
        # target_cps = data['target_cps'].reshape(-1, 6, 3)
        # if len(self.og_grasps.shape) == 4:
        #     self.og_grasps = self.og_grasps.reshape(-1, 4, 4)
        # mlab.figure(bgcolor=(1, 1, 1))
        # draw_scene(
        #         input_pcs[0].cpu().detach().numpy(),
        #         grasps=self.og_grasps,
        #         # target_cps=target_cps,
        #     )
        # mlab.show()
        # print(xd)
        ###END OF CODE SNIPPET

        ###CODE SNIPPET TO VISUALIZE GOOD AND BAD GRASPS FOR THE EVALUATOR
        # self.og_grasps = data['good_og_grasps']
        # #reshape from 32 x 2 x 4 x 4 to 64 x 4 x 4
        # if len(self.og_grasps.shape) == 4:
        #     self.og_grasps = self.og_grasps.reshape(-1, 4, 4)
        # mlab.figure(bgcolor=(1, 1, 1))
        # draw_scene(
        #         input_pcs[0].cpu().detach().numpy(),
        #         grasps=self.og_grasps,
        #     )
        # mlab.show()
        # # Cear the scene
        # # mlab.clf()
        # mlab.figure(bgcolor=(1, 1, 1))
        # self.og_grasps = data['bad_og_grasps']
        # # print(self.og_grasps.shape)
        # #reshape from 32 x 2 x 4 x 4 to 64 x 4 x 4
        # if len(self.og_grasps.shape) == 4:
        #     self.og_grasps = self.og_grasps.reshape(-1, 4, 4)
        # draw_scene(
        #         input_pcs[0].cpu().detach().numpy(),
        #         grasps=self.og_grasps,
        #     )
        # mlab.show()
        # print(xd)
        ###END OF CODE SNIPPET
     
        
        if self.opt.arch == "evaluator":
            targets = torch.from_numpy(data['labels']).float()
            

        else:
            targets = torch.from_numpy(data['target_cps']).float()
        self.pcs = input_pcs.to(self.device).requires_grad_(self.is_train)
        self.grasps = input_grasps.to(self.device).requires_grad_(
            self.is_train)
        self.targets = targets.to(self.device)

    def generate_grasps(self, pcs, z=None):
        with torch.no_grad():
            return self.net.module.generate_grasps(pcs, z=z)

    def evaluate_grasps(self, pcs, gripper_pcs):
        success, _ = self.net.module(pcs, gripper_pcs)
        # success = torch.tensor_split(success, 2, dim=0)
        
        # success = success[0].add(success[1])
        # #Devide all values by 2
        # success = success / 2.0
        # print(torch.sigmoid(success))
        # print(success.shape)
        return torch.sigmoid(success)

    def forward(self):
        return self.net(self.pcs, self.grasps, train=self.is_train)

    def backward(self, out):
        if self.opt.arch == 'vae':
            if self.opt.split_decoders:

                predicted_cp, confidence, mu, logvar = out
                predicted_cp_left = utils.transform_control_points(
                    predicted_cp[0], predicted_cp[0].shape[0], device=self.device, dual_grasp = False)
                predicted_cp_right = utils.transform_control_points(
                    predicted_cp[1], predicted_cp[1].shape[0], device=self.device, dual_grasp = False)
                predicted_cp = [predicted_cp_left, predicted_cp_right]
                self.reconstruction_loss, self.confidence_loss = self.criterion[1](
                    predicted_cp,
                    self.targets,
                    confidence=confidence,
                    confidence_weight=self.opt.confidence_weight,
                    device=self.device,
                    dual_grasp=self.dual_grasp)
                self.kl_loss = self.criterion[0](
                    mu, logvar, device=self.device, dual_grasp=self.dual_grasp)
                self.loss_left = (self.kl_loss[0] * self.opt.kl_loss_weight) + self.reconstruction_loss[0] + self.confidence_loss[0]
                self.loss_right = (self.kl_loss[1] * self.opt.kl_loss_weight) + self.reconstruction_loss[1] + self.confidence_loss[1]

                self.loss = torch.mean(self.loss_left + self.loss_right)
            else:
                predicted_cp, confidence, mu, logvar = out
                #reshape into bsx14 from bsx7
                predicted_cp = torch.cat((predicted_cp[0], predicted_cp[1]), -1)
                mu = torch.cat(mu, -1)
                logvar = torch.cat(logvar, -1)
                confidence = torch.stack(confidence, 1)
                predicted_cp = utils.transform_control_points(
                    predicted_cp, predicted_cp.shape[0], device=self.device, dual_grasp = self.dual_grasp)
                self.reconstruction_loss, self.confidence_loss = self.criterion[1](
                    predicted_cp,
                    self.targets,
                    confidence=confidence,
                    confidence_weight=self.opt.confidence_weight,
                    device=self.device,
                    dual_grasp=self.dual_grasp)
                self.kl_loss = self.opt.kl_loss_weight * self.criterion[0](
                    mu, logvar, device=self.device, dual_grasp=self.dual_grasp)
                self.loss = self.kl_loss + self.reconstruction_loss + self.confidence_loss
        elif self.opt.arch == 'gan':
            if self.opt.split_decoders:
                predicted_cp, confidence = out
                predicted_cp_left = utils.transform_control_points(
                    predicted_cp[0], predicted_cp[0].shape[0], device=self.device, dual_grasp = False)
                predicted_cp_right = utils.transform_control_points(
                    predicted_cp[1], predicted_cp[1].shape[0], device=self.device, dual_grasp = False)
                predicted_cp = [predicted_cp_left, predicted_cp_right]
                self.reconstruction_loss, self.confidence_loss = self.criterion(
                    predicted_cp,
                    self.targets,
                    confidence=confidence,
                    confidence_weight=self.opt.confidence_weight,
                    device=self.device)
                self.loss_left = self.reconstruction_loss[0] + self.confidence_loss[0]
                self.loss_right = self.reconstruction_loss[1] + self.confidence_loss[1]
                self.loss = torch.mean(self.loss_left + self.loss_right)
            else:
                predicted_cp, confidence = out
                #reshape into bsx14 from bsx7
                predicted_cp = torch.cat((predicted_cp[0], predicted_cp[1]), -1)
                confidence = torch.stack(confidence, 1)
                predicted_cp = utils.transform_control_points(
                    predicted_cp, predicted_cp.shape[0], device=self.device)
                self.reconstruction_loss, self.confidence_loss = self.criterion(
                    predicted_cp,
                    self.targets,
                    confidence=confidence,
                    confidence_weight=self.opt.confidence_weight,
                    device=self.device)
                self.loss = self.reconstruction_loss + self.confidence_loss
        elif self.opt.arch == 'evaluator':
            grasp_classification, confidence = out
            self.classification_loss, self.confidence_loss = self.criterion(
                grasp_classification.squeeze(),
                self.targets,
                confidence,
                self.opt.confidence_weight,
                device=self.device)
            self.loss = self.classification_loss + self.confidence_loss
        
        # print(self.loss)
        ###CODE SNIPPET TO VISUALIZE THE PREDICTED CONTROL POINTS
        # # if len(self.og_grasps.shape) == 4:
        # #     self.og_grasps = self.og_grasps.reshape(-1, 4, 4)
        # predicted_cp = torch.cat((predicted_cp[0], predicted_cp[1]), -1)
        # if len(predicted_cp.shape) == 4:
        #     predicted_cp = predicted_cp.reshape(-1, 6, 3)
        # # print(self.og_grasps.shape, predicted_cp.shape)
        # # print(predicted_cp.shape)

        # mlab.figure(bgcolor=(1, 1, 1))
        # draw_scene(
        #         self.pc_for_visualization[0].cpu().detach().numpy(),
        #         # grasps=self.og_grasps,
        #         target_cps=predicted_cp,
        #     )
        # mlab.show()
        # print(xd)
        ##END OF CODE SNIPPET
        if self.opt.split_decoders:
            return self.loss_left, self.loss_right
        else:
            self.loss.backward()

    def optimize_parameters(self):
        if self.opt.split_decoders:
            self.optimizer_left.zero_grad()
            self.optimizer_right.zero_grad()
            out = self.forward()
            loss_left, loss_right = self.backward(out)
            loss_left.backward()
            self.optimizer_left.step()
            loss_right.backward()
            self.optimizer_right.step()
        else:
            self.optimizer.zero_grad()
            out = self.forward()
            self.backward(out)
            self.optimizer.step()


##################

    def load_network(self, which_epoch, train=True):
        """load model from disk"""
        save_filename = '%s_net.pth' % which_epoch
        load_path = join(self.save_dir, save_filename)
        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        checkpoint = torch.load(load_path, map_location=self.device)
        if hasattr(checkpoint['model_state_dict'], '_metadata'):
            del checkpoint['model_state_dict']._metadata
        net.load_state_dict(checkpoint['model_state_dict'])
        if train and self.opt.split_decoders:
            self.optimizer_left.load_state_dict(checkpoint['optimizer_left_state_dict'])
            self.optimizer_right.load_state_dict(checkpoint['optimizer_right_state_dict'])
            self.scheduler_left.load_state_dict(checkpoint['scheduler_left_state_dict'])
            self.scheduler_right.load_state_dict(checkpoint['scheduler_right_state_dict'])
            self.opt.epoch_count = checkpoint["epoch"]
        elif train:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.opt.epoch_count = checkpoint["epoch"]
        else:
            net.eval()

    def save_network(self, net_name, epoch_num):
        """save model to disk"""
        save_filename = '%s_net.pth' % (net_name)
        save_path = join(self.save_dir, save_filename)
        if self.split_decoders:
            torch.save(
                {
                    'epoch': epoch_num + 1,
                    'model_state_dict': self.net.module.cpu().state_dict(),
                    'optimizer_left_state_dict': self.optimizer_left.state_dict(),
                    'optimizer_right_state_dict': self.optimizer_right.state_dict(),
                    'scheduler_left_state_dict': self.scheduler_left.state_dict(),
                    'scheduler_right_state_dict': self.scheduler_right.state_dict(),
                }, save_path)
        else:
            torch.save(
                {
                    'epoch': epoch_num + 1,
                    'model_state_dict': self.net.module.cpu().state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                }, save_path)

        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            self.net.cuda(self.gpu_ids[0])

    def update_learning_rate(self):
        """update learning rate (called once every epoch)"""
        if self.split_decoders:
            self.scheduler_left.step()
            lr_left = self.optimizer_left.param_groups[0]['lr']
            self.scheduler_right.step()
            lr_right = self.optimizer_right.param_groups[0]['lr']
            print('learning rate = %.7f' % lr_left)
        else:
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']
            print('learning rate = %.7f' % lr)

    def test(self):
        """tests model
        returns: number correct and total number
        """
        with torch.no_grad():
            out = self.forward()
            prediction, confidence = out
            
            if self.opt.arch == "vae":
                if self.opt.split_decoders:
                    predicted_cp_left = utils.transform_control_points(
                        prediction[0], prediction[0].shape[0], device=self.device, dual_grasp = False)
                    predicted_cp_right = utils.transform_control_points(
                        prediction[1], prediction[1].shape[0], device=self.device, dual_grasp = False)
                    predicted_cp = [predicted_cp_left, predicted_cp_right]
                    reconstruction_loss, confidence_loss = self.criterion[1](
                        predicted_cp,
                        self.targets,
                        confidence=confidence,
                        confidence_weight=self.opt.confidence_weight,
                        device=self.device,
                        dual_grasp=self.dual_grasp)
                    return torch.mean(torch.stack(reconstruction_loss)), 1
                else:
                    # predicted_cp, confidence, mu, logvar = out
                    #reshape into bsx14 from bsx7
                    predicted_cp = torch.cat((prediction[0], prediction[1]), -1)
                    confidence = torch.stack(confidence, 1)
                    predicted_cp = utils.transform_control_points(
                        predicted_cp, predicted_cp.shape[0], device=self.device, dual_grasp = self.dual_grasp)
                    reconstruction_loss, confidence_loss = self.criterion[1](
                        predicted_cp,
                        self.targets,
                        confidence=confidence,
                        confidence_weight=self.opt.confidence_weight,
                        device=self.device,
                        dual_grasp=self.dual_grasp)
                    return reconstruction_loss, 1
            elif self.opt.arch == "gan":
                if self.opt.split_decoders:
                    predicted_cp_left = utils.transform_control_points(
                        prediction[0], prediction[0].shape[0], device=self.device, dual_grasp = False)
                    predicted_cp_right = utils.transform_control_points(
                        prediction[1], prediction[1].shape[0], device=self.device, dual_grasp = False)
                    predicted_cp = [predicted_cp_left, predicted_cp_right]
                    reconstruction_loss, confidence_loss = self.criterion(
                        predicted_cp,
                        self.targets,
                        confidence=confidence,
                        confidence_weight=self.opt.confidence_weight,
                        device=self.device)
                    return torch.mean(torch.stack(reconstruction_loss)), 1

                else:
                    #reshape into bsx14 from bsx7
                    predicted_cp = torch.cat((prediction[0], prediction[1]), -1)
                    confidence = torch.stack(confidence, 1)
                    predicted_cp = utils.transform_control_points(
                        predicted_cp, predicted_cp.shape[0], device=self.device)
                    reconstruction_loss,confidence_loss = self.criterion(
                        predicted_cp,
                        self.targets,
                        confidence=confidence,
                        confidence_weight=self.opt.confidence_weight,
                        device=self.device)
                    return reconstruction_loss, 1
            else:
                # if self.dual_grasp:
                #     predicted = torch.sigmoid(prediction).squeeze()
                #     predicted = torch.tensor_split(predicted, 2, dim=0)
    
                #     predicted = predicted [0].add(predicted [1])
                #     #Devide all values by 2
                #     predicted  = predicted  / 2.0
                #     predicted = torch.round(predicted)
                # else: 
                # print("predicted unrounded", torch.sigmoid(prediction).squeeze())

                predicted = torch.round(torch.sigmoid(prediction)).squeeze()
                # print("predicted rounded", predicted)
                # print("targets", self.targets)
                correct = (predicted == self.targets).sum().item()
                return correct, len(self.targets)
