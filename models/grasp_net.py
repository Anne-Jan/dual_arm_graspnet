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

        if self.is_train:
            self.optimizer = torch.optim.Adam(self.net.parameters(),
                                              lr=opt.lr,
                                              betas=(opt.beta1, 0.999))
            self.scheduler = networks.get_scheduler(self.optimizer, opt)
        if not self.is_train or opt.continue_train:
            self.load_network(opt.which_epoch, self.is_train)

    def set_input(self, data):
        input_pcs = torch.from_numpy(data['pc']).contiguous()
        self.pc_for_visualization = input_pcs
        input_grasps = torch.from_numpy(data['grasp_rt']).float()
        #check if the input grasps and pc are not nan
        if torch.isnan(input_pcs).any() or torch.isnan(input_grasps).any():
            print("Input pc or grasps are nan")
        self.og_grasps = data['og_grasps']
        # print("input_grasps", input_grasps.shape)
        target_cps = data['target_cps'].reshape(-1, 6, 3)
        # print(len(self.og_grasps))
        # reshape from 32 x 2 x 4 x 4 to 64 x 4 x 4
        # if len(self.og_grasps.shape) == 4:
        #     self.og_grasps = self.og_grasps.reshape(-1, 4, 4)
        # mlab.figure(bgcolor=(1, 1, 1))
        # draw_scene(
        #         input_pcs[0].cpu().detach().numpy(),
        #         grasps=self.og_grasps,
        #         target_cps=target_cps,
        #     )
        # mlab.show()
        # print(xd)

        ###CODE SNIPPET TO VISUALIZE GOOD AND BAD GRASPS FOR THE EVALUATOR
        # self.og_grasps = data['good_og_grasps']
        # print(len(self.og_grasps))
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
        # mlab.clf()
        # mlab.figure(bgcolor=(1, 1, 1))
        # self.og_grasps = data['bad_og_grasps']
        # print(self.og_grasps.shape)
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
        return torch.sigmoid(success)

    def forward(self):
        return self.net(self.pcs, self.grasps, train=self.is_train)

    def backward(self, out):
        if self.opt.arch == 'vae':
            predicted_cp, confidence, mu, logvar = out
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
            # print(self.reconstruction_loss, self.confidence_loss, self.kl_loss)
            #catch if any of the losses are nan
            # print(predicted_cp[:,:,0].min(), predicted_cp[:,:,0].max())
            # print(predicted_cp[:,1,:,0].min(), predicted_cp[:,1,:,0].max())
            # print(predicted_cp[:,:,1].min(), predicted_cp[:,:,1].max())
            # print(predicted_cp[:,1,:,1].min(), predicted_cp[:,1,:,1].max())
            # print(predicted_cp[:,:,2].min(), predicted_cp[:,:,2].max())
            # print(predicted_cp[:,1,:,2].min(), predicted_cp[:,1,:,2].max())
            # print()
            if torch.isnan(self.reconstruction_loss) or torch.isnan(self.confidence_loss) or torch.isnan(self.kl_loss):
                print("Loss is nan")
                print("Reconstruction Loss: ", self.reconstruction_loss, "Confidence Loss: ", self.confidence_loss, "KL Loss: ", self.kl_loss)
            self.loss = self.kl_loss + self.reconstruction_loss + self.confidence_loss
        elif self.opt.arch == 'gan':
            predicted_cp, confidence = out
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
        

        ###CODE SNIPPET TO VISUALIZE THE PREDICTED CONTROL POINTS
        # if len(self.og_grasps.shape) == 4:
        #     self.og_grasps = self.og_grasps.reshape(-1, 4, 4)
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
        self.loss.backward()

    def optimize_parameters(self):
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
        if train:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.opt.epoch_count = checkpoint["epoch"]
        else:
            net.eval()

    def save_network(self, net_name, epoch_num):
        """save model to disk"""
        save_filename = '%s_net.pth' % (net_name)
        save_path = join(self.save_dir, save_filename)
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
                # print("DUAL GRASP", self.dual_grasp)
                predicted_cp = utils.transform_control_points(
                    prediction, prediction.shape[0], device=self.device, dual_grasp = self.dual_grasp)
                # print(predicted_cp.shape[0])
                reconstruction_loss, _ = self.criterion[1](
                    predicted_cp,
                    self.targets,
                    confidence=confidence,
                    confidence_weight=self.opt.confidence_weight,
                    device=self.device,
                    dual_grasp=self.dual_grasp)
                return reconstruction_loss, 1
            elif self.opt.arch == "gan":
                predicted_cp = utils.transform_control_points(
                    prediction, prediction.shape[0], device=self.device)
                reconstruction_loss, _ = self.criterion(
                    predicted_cp,
                    self.targets,
                    confidence=confidence,
                    confidence_weight=self.opt.confidence_weight,
                    device=self.device)
                return reconstruction_loss, 1
            else:
                if self.dual_grasp:
                    predicted = torch.sigmoid(prediction).squeeze()
                    predicted = torch.tensor_split(predicted, 2, dim=0)
    
                    predicted = predicted [0].add(predicted [1])
                    #Devide all values by 2
                    predicted  = predicted  / 2.0
                    predicted = torch.round(predicted)
                else: 
                    predicted = torch.round(torch.sigmoid(prediction)).squeeze()
               
                correct = (predicted == self.targets).sum().item()
                return correct, len(self.targets)
