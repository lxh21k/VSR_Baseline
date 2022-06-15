import torch
from models.edvr_net import EDVR
from torch.nn.parallel import DataParallel, DistributedDataParallel

class EDVRModel():
    """EDVR Model.

    Paper: EDVR: Video Restoration with Enhanced Deformable Convolutional Networks.  # noqa: E501
    """

    def __init__(self, params):
        super(EDVRModel, self).__init__()
        self.is_train = params.is_train
        if self.is_train:
            self.train_tsa_iter = params.train['tsa_iter']


        self.device = torch.device('cuda' if params.num_gpu != 0 else 'cpu')
        self.net = EDVR(num_in_ch=params.network['num_in_ch'],
                        num_out_ch=params.network['num_out_ch'],
                        num_feat=params.network['num_feat'],
                        num_frame=params.network['num_frame'],
                        deformable_groups=params.network['deformable_groups'],
                        num_extract_block=params.network['num_extract_block'],
                        num_reconstruct_block=params.network['num_reconstruct_block']
                        )

        print(self.device == "cuda")

        if params.cuda:
            print("Loading model to GPU...")
            self.net = self.net.cuda()
            self.net = DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

        self.print_network(self.net)

    def setup_optimizers(self):
        train_opt = self.params['train']
        dcn_lr_mul = train_opt.get('dcn_lr_mul', 1)
        # logger = get_root_logger()
        # logger.info(f'Multiple the learning rate for dcn with {dcn_lr_mul}.')
        if dcn_lr_mul == 1:
            optim_params = self.net_g.parameters()
        else:  # separate dcn params and normal params for different lr
            normal_params = []
            dcn_params = []
            for name, param in self.net_g.named_parameters():
                if 'dcn' in name:
                    dcn_params.append(param)
                else:
                    normal_params.append(param)
            optim_params = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': train_opt['optim_g']['lr']
                },
                {
                    'params': dcn_params,
                    'lr': train_opt['optim_g']['lr'] * dcn_lr_mul
                },
            ]

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def optimize_parameters(self, current_iter):
        if self.train_tsa_iter:
            if current_iter == 1:
                # logger = get_root_logger()
                # logger.info(f'Only train TSA module for {self.train_tsa_iter} iters.')
                for name, param in self.net_g.named_parameters():
                    if 'fusion' not in name:
                        param.requires_grad = False
            elif current_iter == self.train_tsa_iter:
                # logger = get_root_logger()
                # logger.warning('Train all the parameters.')
                for param in self.net_g.parameters():
                    param.requires_grad = True

        super(EDVRModel, self).optimize_parameters(current_iter)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def get_bare_model(self, net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net


    def print_network(self, net):
        """Print the str and parameter number of a network.

        Args:
            net (nn.Module)
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net_cls_str = f'{net.__class__.__name__} - {net.module.__class__.__name__}'
        else:
            net_cls_str = f'{net.__class__.__name__}'

        net = self.get_bare_model(net)
        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))

        # logger = get_root_logger()
        print(f'Network: {net_cls_str}, with parameters: {net_params:,d}')
        print(net_str)