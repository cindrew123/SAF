import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
from ..util import util as util
import torch.nn as nn
import torch.nn.functional as F
style_Innovation = 1# 1:OPEN the trans_style in STS,0 no.
IGNORE_LABEL = 255
Reinforcement_Learning = "FALSE" # "TRUE","FALSE"
memory_out_switch=1 # 0, the memory result is not input into semantic segmentation

class STSMODEL(BaseModel):
    """ This class implements STS and FastSTS model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for STS model
        """
        parser.add_argument('--STS_mode', type=str, default="STS", choices='(STS, STS, FastSTS, fastSTS)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='reshape', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastSTS, but not STS")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for STS and FastSTS
        if opt.STS_mode.lower() == "STS":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.STS_mode.lower() == "fastSTS":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.STS_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        # Ensure requires_grad is set correctly for network parameters
            for param in self.netG.parameters():
                param.requires_grad = True
            for param in self.netF.parameters():
                param.requires_grad = True
            for param in self.netD.parameters():
                param.requires_grad = True
                
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        self.conv_layer_visuals = nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1).cuda()

        self.conv_fusion = nn.Sequential(
            nn.Conv2d(9, 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3, padding=1)
        ).cuda()
        self.attention_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        ).cuda()
        self.conv_Gradient_fusion = nn.Sequential(
            nn.Conv2d(3 * 2, 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3, padding=1)
        ).cuda()


    def data_dependent_initialize(self, data1, data2,pred_labels):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data1,data2)
        self.real_A.requires_grad_(True)
        self.real_B.requires_grad_(True)
        self.get_fake_imgs(data2)
        self.set_predicted_labels(pred_labels)
        self.forward()                     # compute fake images: G(A)
        print ("inital again")
        print("isTrain is ", self.opt.isTrain)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G

            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)


    def print_gradients(self, model):

        print("Gradients for each layer:")
        for name, param in model.named_parameters():
            if 'transformer' not in name:
                if param.grad is not None:
                    print(f"Layer: {name} | requires_grad: {param.requires_grad}")
                    if param.grad.dim() == 4:
                        print(f"Layer: {name} | Gradient: {param.grad[0, 0, :, :]}")
                    elif param.grad.dim() == 3:
                        print(f"Layer: {name} | Gradient (sampled): {param.grad[0, :, :]}")
                    elif param.grad.dim() == 2:
                            print(f"Layer: {name} | Gradient (sampled): {param.grad[:, :]}")
                    else:
                            print(f"Layer: {name} | Gradient (full): {param.grad}")
                else:
                    print(f"Layer: {name} | Gradient: None")

    def optimize_parameters(self):
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()
        #self.print_gradients(self.netD)
        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()


        # Reinforcement Learning.  Reward design: if the discriminator thinks the generated image is real (value close to 1), the generator's reward is higher
        if Reinforcement_Learning == "TRUE":
            pred_fake = self.netD(self.fake_B)
            reward = torch.mean(pred_fake).item() #Calculate the average output of the discriminator as the reward
            adjusted_loss_G = self.loss_G * (1 + reward)
            adjusted_loss_G.backward()
        else:
            self.loss_G.backward(retain_graph=True)
        #self.print_gradients(self.netG)

        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, real_A, real_B, PTS_labels=None):
        """Set the input images directly.
        Parameters:
            real_A (tensor): source domain image
            real_B (tensor): target domain ima
        """
        self.real_A = real_A
        self.real_B = real_B
        if PTS_labels is not None:
            self.PTS_labels = PTS_labels
        # Ensure the input dimensions are correct (4D tensor: [batch_size, channels, height, width])
        if len(self.real_A.shape) == 3:
            self.real_A = self.real_A.unsqueeze(0)
        if len(self.real_B.shape) == 3:
            self.real_B = self.real_B.unsqueeze(0)



    def get_generated_image(self,style_Innovation):
        """Return the generated fake images."""
        if style_Innovation == 1:
            # return self.fake_B,self.fake_B_tran,self.fake_B2A ,self.fake_B2A_tran,self.optimized_fake
            return self.fake_B, self.fake_B_tran, self.memory_out
        else:
            return self.fake_B

    def get_style_layers(self):
        return self.content_layers_output,self.style_layers_output

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])
        if style_Innovation == 1:
            self.fake,self.trasout_adaptive,self.memory_out,self.content_layers_output,self.style_layers_output,self.Memory_Affinity,self.memory_current_feature = self.netG(self.real)
            # self.fake_finall = self.fake[0,:]  # not right
            # self.fake_B is a fake img from ATOB
            self.fake_B = self.fake[:self.real_A.size(0)] # [1, 3, 512, 512]
            self.fake_B_tran = self.trasout_adaptive[:self.real_A.size(0)] # [1, 3, 512, 512]

            # self.fake_B2A is a fake img from bTOa
            start_idx = self.real_A.size(0)
            end_idx = start_idx + self.real_B.size(0)
            self.real_A_tran = self.trasout_adaptive[start_idx:end_idx] # [1, 3, 512, 512] # Initial image semantic perception
        else:
            self.fake = self.netG(self.real)
            self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]



    def output_loss(self):
        return self.loss_G_GAN, self.Contrastive_loss, self.semantic_loss, self.semantic_perception_loss, self.loss_G, self.loss_D,self.memory_loss

    def compute_memory_loss(self,current_features, retrieved_memory_features):
        """
        Calculate memory loss to make the current feature map consistent with the retrieved memory feature map.
        Parameters:
        - current_features: feature map of the current generated image
        - retrieved_memory_features: the most similar memory feature map retrieved from the memory module

        Returns:
        - loss: memory loss value
        """
        # Use mean square error (MSE) loss to measure the gap between the current feature map and the memory feature map
        self.loss = F.mse_loss(current_features, retrieved_memory_features)
        return self.loss

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        self.Contrastive_loss = self.calculate_Contrastive_loss(self.real_A.detach(), self.real_B.detach(), self.fake_B.detach())
        # First, G(A) should fake the discriminator

        pred_fake = self.netD(fake)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN

        #self.semantic_perception_loss = self.compute_semantic_perception_loss(self.real_A_tran, self.fake_B_tran)
        # Style semantic consistency loss
        self.semantic_loss = self.compute_semantic_consistency_loss(self.fake_B_tran, self.PTS_labels)

        # Compute semantic perception loss
        self.semantic_perception_loss = self.compute_semantic_perception_loss(self.real_A_tran, self.fake_B_tran)
        if memory_out_switch == 1:
            self.memory_loss = 0.01*self.compute_memory_loss(self.memory_current_feature,self.Memory_Affinity)
        else:
            self.memory_loss = 0.0

        '''
        print ("semantic_perception_loss is ", 5*self.semantic_perception_loss)
        print("loss_G_GAN is ", self.loss_G_GAN)
        print ("semantic_loss is ", 0.1*self.semantic_loss)
        print("Contrastive_loss is ", self.Contrastive_loss)
        print("memory_loss is ", self.memory_loss)
        '''

        self.loss_G = self.loss_G_GAN + self.Contrastive_loss +  self.semantic_loss+ 5*self.semantic_perception_loss+self.memory_loss
        #self.loss_G = self.loss_G_GAN
        return self.loss_G

    def compute_semantic_perception_loss(self, real_A_tran, fake_B_tran):
        """
        Calculate the semantic perception loss using real_A_tran and fake_B_tran.
        This loss enforces semantic consistency between real_A and the generated fake_B.

        Parameters:
        - real_A_tran (tensor): The semantic features of real_A.
        - fake_B_tran (tensor): The semantic features of fake_B (generated from real_A).

        Returns:
        - loss (tensor): The calculated semantic perception loss.
        """
        # Ensure both feature maps have the same size
        assert real_A_tran.shape == fake_B_tran.shape, "Feature map sizes do not match"

        # Calculate the semantic perception loss (MSE or L1 loss)
        loss = torch.nn.functional.mse_loss(real_A_tran, fake_B_tran)

        return loss

    def get_fake_imgs(self, img):
        """
        Set the predicted labels for the generated image PTS.

        Parameters:
        - pred_labels (tensor): Predicted labels for PTS, shape (batch_size, height, width).
        """
        self.generate_fake_img = img

    def set_predicted_labels(self, pred_labels):
        """
        Set the predicted labels for the generated image PTS.

        Parameters:
        - pred_labels (tensor): Predicted labels for PTS, shape (batch_size, height, width).
        """
        self.pred_labels = pred_labels.clone().detach()

    def compute_semantic_consistency_loss(self, PTS, PTS_labels):
        """
        Calculate the semantic consistency loss using the generated image PTS and its labels.
        """
        if PTS.dtype != torch.float32:
            PTS = PTS.float()
        if PTS_labels.dtype != torch.long:
            PTS_labels = PTS_labels.long()
        if PTS.dim() != 4:
            return torch.tensor(0.0, device=PTS.device, requires_grad=True)
        else:
            if PTS_labels.dim() == 2:
                PTS_labels = PTS_labels.unsqueeze(0)
            criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL).cuda()
            loss = criterion(PTS, PTS_labels)
            return loss


    def calculate_Contrastive_loss(self, real_A,real_B, fake_B):
        # Get feature representation
        feat_real_A = self.netG(real_A, self.nce_layers, encode_only=True)
        feat_real_B = self.netG(real_B, self.nce_layers, encode_only=True)
        feat_fake_B = self.netG(fake_B, self.nce_layers, encode_only=True)
        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_fake_B = [torch.flip(fg, [3]) for fg in feat_fake_B]
            feat_real_A = [torch.flip(fa, [3]) for fa in feat_real_A]
            feat_real_B = [torch.flip(fb, [3]) for fb in feat_real_B]
        feat_fake_B_pool, sample_ids = self.netF(feat_fake_B, self.opt.num_patches, None)
        feat_fake_real_A_pool, _ = self.netF(feat_real_A, self.opt.num_patches, sample_ids)
        feat_real_B_pool, _ = self.netF(feat_real_B, self.opt.num_patches, sample_ids)

        # Calculate positive sample contrast loss
        pos_loss = 0.0
        for f_tgt, f_pos in zip(feat_fake_B_pool, feat_real_B_pool):
            pos_loss += torch.nn.functional.mse_loss(f_tgt, f_pos)
        # Calculate negative sample contrast loss
        neg_loss  = 0.0
        for f_tgt, f_src in zip(feat_fake_B_pool, feat_fake_real_A_pool):
            neg_loss  += torch.nn.functional.mse_loss(f_tgt, f_src)
        margin = 0.1
        total_loss = torch.relu(pos_loss - neg_loss + margin)
        return total_loss

    def Semantic_perception(self,fake_B,fake_B_tran,style_images,Style_Control_Category ):
        # !!!!!!!!!Do not change any parameters or the final processing method of the image. This is the optimal combination obtained after many experiments.
        fake_B_tran = torch.sigmoid(fake_B_tran)
        # best one is Semantic_Awareness_Multiplication.because Multiplication is best way to  Connect pixels with strong internal semantic information
        if Style_Control_Category == "Semantic_Awareness_Multiplication":
            mask = torch.cat((fake_B, fake_B_tran), dim=1)
            mask = self.conv_layer_visuals(mask)
            fake_B_tran = mask*fake_B
            alpha = 0.1 # only 0.1
            STS_visuals = fake_B_tran * alpha + fake_B
            # make fake_B_tran_mapped * rate + fake_B to rate*fake_B.max()-rate*fake_B.min()
            STS_visuals = self.match_histogram_std(style_images, STS_visuals) # do not change
        # trans as semantic guidance to generate images with stronger semantic connections
        if Style_Control_Category == "Semantic_Awareness_Addition":
            STS_visuals = torch.cat((fake_B, fake_B_tran), dim=1)
            STS_visuals = self.conv_fusion(STS_visuals)
            alpha = 0.1 # only 0.1
            STS_visuals =  alpha * STS_visuals + (1 - alpha) * fake_B
            STS_visuals = self.match_histogram_std(style_images, STS_visuals) # do not change
        # Using attention maps to guide semantic perception
        if Style_Control_Category == "Attention_Semantic_Awareness":
            attention_weights = self.attention_net(fake_B_tran)
            attention_weights = attention_weights.repeat(1, fake_B.size(1), 1, 1)
            STS_visuals = attention_weights * fake_B_tran + (1 - attention_weights) * fake_B
            STS_visuals = self.conv_fusion(torch.cat((STS_visuals, fake_B), dim=1))
            alpha = 0.2 # 0.2 and 0.1 ok
            STS_visuals = alpha * STS_visuals + (1 - alpha) * fake_B
            STS_visuals = self.match_histogram_std(style_images, STS_visuals) # do not change
        return STS_visuals

    def match_histogram_simple(self, style_images, STS_visuals):
        style_max, style_min = style_images.max(), style_images.min()
        STS_max, STS_min = STS_visuals.max(), STS_visuals.min()
        normalized_STS_visuals = (STS_visuals - STS_min) / (STS_max - STS_min + 1e-8)
        matched_visuals = normalized_STS_visuals * (style_max - style_min) + style_min
        matched_visuals = torch.clamp(matched_visuals, min=style_min.item(), max=style_max.item())

        return matched_visuals

    def match_histogram_std(self,source, target): # img target-->same as img source
        source_mean, source_std = torch.mean(source), torch.std(source)
        target_mean, target_std = torch.mean(target), torch.std(target)
        source_max, source_min = source.max(), source.min()
        matched = (target - target_mean) / (target_std + 1e-8) * (source_std + 1e-8) + source_mean
        matched = torch.clamp(matched, min=source_min.item(), max=source_max.item())
        return matched

    def match_histogram_max(self, source, target): # img target-->same as img source
        source_mean, source_std = torch.mean(source), torch.std(source)
        target_mean, target_std = torch.mean(target), torch.std(target)

        matched = (target - target_mean) / (target_std + 1e-8) * (source_std + 1e-8) + source_mean

        source_max, source_min = source.max(), source.min()
        target_max, target_min = target.max(), target.min()

        target_range = target_max - target_min
        source_range = source_max - source_min

        matched = (matched - target_min) / (target_range + 1e-8) * (source_range + 1e-8) + source_min
        matched = torch.clamp(matched, min=source_min.item(), max=source_max.item())
        return matched


    def Style_Gradient(self,fake_B,style_A, style_B,Style_Gradient_Control_Category,alpha ):
        # Attention-map guided style gradient control
        if Style_Gradient_Control_Category == "attention":
            if len(style_A.size()) == 3:
                style_A= style_A.unsqueeze(0)
            attention_map = self.attention_net(fake_B)
            attention_map = attention_map.repeat(1, style_A.size(1), 1, 1)
            style_weight_A = 1 - attention_map
            style_weight_B = alpha * attention_map
            blended_style = style_weight_A * style_A + style_weight_B * fake_B
            #blended_style = torch.clamp(blended_style, min=0, max=alpha)
            STS_visuals =  (1 - alpha) * style_A + alpha * fake_B
            STS_visuals = (1 - alpha) * blended_style + alpha * STS_visuals
            STS_visuals = self.match_histogram_std(style_B, STS_visuals)
        # Simple gradient control
        if Style_Gradient_Control_Category == "Interpolation":
            if len(style_A.size()) == 3:
                style_A= style_A.unsqueeze(0)
            STS_visuals = (1 - alpha) * style_A + alpha * fake_B
            STS_visuals = self.match_histogram_simple(style_B, STS_visuals) #use different match_histogram to change the final img
        return STS_visuals