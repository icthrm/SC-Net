from dataset_EM import *
from model_3d import *
import torch.nn as nn
from gaussian_filter import GaussianDenoise3d
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from loss import gradient, mean_by_blocks
from utils import *
import cv2, mrc, datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sampling(tomo, scale, mode='nearest'):
    tomo = torch.from_numpy(tomo)
    tomo = tomo.unsqueeze(0).unsqueeze(0)
    tomo = F.interpolate(tomo, size=(int(tomo.shape[2] * scale), tomo.shape[3], int(tomo.shape[4] * scale)),
                         mode=mode)
    tomo = tomo.squeeze(0).squeeze(0)
    tomo = np.array(tomo)
    return tomo


def append_index(dir_result, fileset, step=False):
    index_path = os.path.join(dir_result, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        for key, value in fileset.items():
            index.write("<th>%s</th>" % key)
        index.write('</tr>')

    # for fileset in filesets:
    index.write("<tr>")

    if step:
        index.write("<td>%d</td>" % fileset["step"])
    index.write("<td>%s</td>" % fileset["name"])

    del fileset['name']

    for key, value in fileset.items():
        index.write("<td><img src='images/%s'></td>" % value)

    index.write("</tr>")
    return index_path


class Train:
    def __init__(self, args):
        self.mode = args.mode
        self.train_continue = args.train_continue
        self.norm = args.norm

        # file directories
        self.dir_checkpoint = args.dir_checkpoint
        self.dir_log = args.dir_log
        self.name_data = args.name_data
        self.dir_data = args.dir_data
        self.dir_result = args.dir_result
        self.dir_test_tomo = args.dir_test_tomo

        # hyper parameters
        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size
        self.lr_G = args.lr_G
        self.optim = args.optim
        self.beta1 = args.beta1

        # CAUTION: ET image Data Format: C-ZYX, corresponding with this order:
        # (nch_in, ny_in, nx_in), size of original input images without cropping patches
        self.ny_in = args.ny_in
        self.nx_in = args.nx_in
        self.nch_in = args.nch_in

        # (nch_load, ny_load, nx_load), size of input images patches
        self.ny_load = args.ny_load
        self.nx_load = args.nx_load
        self.nch_load = args.nch_load

        # (nch_out, ny_out, nx_out), size of predicted patches
        self.ny_out = args.ny_out
        self.nx_out = args.nx_out
        self.nch_out = args.nch_out

        # number of kernels
        self.nch_ker = args.nch_ker

        # data type for images, default as np.float32
        self.data_type = args.data_type

        self.num_freq_disp = args.num_freq_disp
        self.num_freq_save = args.num_freq_save

        self.gpu_ids = args.gpu_ids

        # newly added params for PriorFusion
        self.tilesize = args.tilesize
        self.swap_window_size = args.swap_window_size
        self.swap_mode = args.swap_mode
        self.swap_ratio = args.swap_ratio
        self.swap_region = args.swap_region
        self.swap_size = args.swap_size
        self.padding = args.padding
        self.N_train = args.N_train
        self.N_test = args.N_test
        self.lambda_exp = args.lambda_exp
        self.lambda_grad = args.lambda_grad
        self.lambda_TV = args.lambda_TV
        self.lambda_median = args.lambda_median

        if self.gpu_ids and torch.cuda.is_available():
            self.device = torch.device("cuda:%d" % self.gpu_ids[0])
            torch.cuda.set_device(self.gpu_ids[0])
        else:
            self.device = torch.device("cpu")

    def save(self, dir_chck, netG, optimG, epoch):
        if not os.path.exists(dir_chck):
            os.makedirs(dir_chck)

        torch.save({'netG': netG.state_dict(),
                    'optimG': optimG.state_dict()},
                   '%s/model_epoch%04d.pth' % (dir_chck, epoch))

    def load(self, dir_chck, netG, optimG=[], epoch=[], mode='train'):

        if not os.path.exists(dir_chck) or not os.listdir(dir_chck):
            epoch = 0
            if mode == 'train':
                return netG, optimG, epoch
            elif mode == 'test':
                return netG, epoch

        if not epoch:
            ckpt = os.listdir(dir_chck)
            ckpt.sort()
            epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

        dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_chck, epoch))

        print('Loaded NO.%d network' % epoch)

        if mode == 'train':
            netG.load_state_dict(dict_net['netG'])
            optimG.load_state_dict(dict_net['optimG'])
            return netG, optimG, epoch

        elif mode == 'test':
            netG.load_state_dict(dict_net['netG'])
            return netG, epoch

    def train(self):
        mode = self.mode

        train_continue = self.train_continue
        num_epoch = self.num_epoch

        lr_G = self.lr_G

        batch_size = self.batch_size
        device = self.device

        gpu_ids = self.gpu_ids

        # essential input for UNET3D Training
        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker

        swap_size = self.swap_size
        tilesize = self.tilesize
        swap_window_size = self.swap_window_size
        swap_mode = self.swap_mode
        swap_ratio = self.swap_ratio
        swap_region = self.swap_region

        lambda_exp = self.lambda_exp
        lambda_grad = self.lambda_grad
        lambda_TV = self.lambda_TV
        lambda_median = self.lambda_median

        norm = self.norm
        name_data = self.name_data

        num_freq_disp = self.num_freq_disp
        num_freq_save = self.num_freq_save

        print('Structural Loss is weighted by '+str(lambda_exp))
        print('Local Gradients Loss is weighted by ' + str(lambda_grad))
        print('Total Variation Loss is weighted by ' + str(lambda_TV))
        print('Mean Constraint Loss is weighted by ' + str(lambda_median))

        ## setup dataset
        dir_chck = os.path.join(self.dir_checkpoint, name_data)

        dir_noisy_train_data = os.path.join(self.dir_data, name_data, 'train_noisy')
        dir_prior_train_data = os.path.join(self.dir_data, name_data, 'train_prior')

        dir_noisy_val_data = os.path.join(self.dir_data, name_data, 'val_noisy')
        dir_prior_val_data = os.path.join(self.dir_data, name_data, 'val_prior')

        dir_log_train = os.path.join(self.dir_log, name_data, 'train')
        dir_log_val = os.path.join(self.dir_log, name_data, 'val')

        dir_result_train = os.path.join(self.dir_result, name_data, 'train')
        dir_result_val = os.path.join(self.dir_result, name_data, 'val')
        if not os.path.exists(os.path.join(dir_result_train, 'images')):
            os.makedirs(os.path.join(dir_result_train, 'images'))
        if not os.path.exists(os.path.join(dir_result_val, 'images')):
            os.makedirs(os.path.join(dir_result_val, 'images'))

        transform_inv = transforms.Compose([ToNumpy(), Denormalize(mean=0.5, std=0.5)])

        dataset_train = Dataset3D(dir_noisy_train_data, dir_prior_train_data, tilesize = tilesize, swap_size = swap_size,
                                  swap_window_size = swap_window_size, swap_mode = swap_mode, swap_region = swap_region,
                                  swap_ratio = swap_ratio, N_train = self.N_train, N_test = self.N_test, mode='train')

        dataset_val = Dataset3D(dir_noisy_val_data, dir_prior_val_data, tilesize = tilesize, swap_size = swap_size,
                                swap_window_size = swap_window_size, swap_mode = swap_mode, swap_region = swap_region,
                                swap_ratio = swap_ratio, N_train = self.N_train, N_test = self.N_test, mode='test')

        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

        num_train = len(dataset_train)
        num_val = len(dataset_val)

        print(str(num_train) +' patches for training in total.')
        print(str(num_val) +' patches for validation in total.')

        num_batch_train = int((num_train / batch_size) + ((num_train % batch_size) != 0))
        num_batch_val = int((num_val / batch_size) + ((num_val % batch_size) != 0))

        ## setup network
        netG = UNet3DEM(nch_in, nch_out, nch_ker, norm)
        netG = nn.DataParallel(netG)
        init_net(netG, init_type='kaiming', init_gain=0.02, gpu_ids=gpu_ids)

        ## setup loss & optimization
        L2_Loss = nn.MSELoss().to(device)     # Regression loss: L2
        L1_Norm_Loss = nn.L1Loss().to(device)
        paramsG = netG.parameters()
        optimG = torch.optim.Adam(paramsG, lr=lr_G, betas=(self.beta1, 0.999))
        # schedulerG = StepLR(optimG, step_size=3, gamma=0.1)
        
        ## load from checkpoints
        st_epoch = 0

        if train_continue == 'on':
            netG, optimG, st_epoch = self.load(dir_chck, netG, optimG, mode=mode)

        ## setup tensorboard
        writer_train = SummaryWriter(log_dir=dir_log_train)
        writer_val = SummaryWriter(log_dir=dir_log_val)

        for epoch in range(st_epoch + 1, num_epoch + 1):
            ## training phase
            netG.train()
            loss_G_train = []

            for batch, data in enumerate(loader_train, 1):
                def should(freq):
                    return freq > 0 and (batch % freq == 0 or batch == num_batch_train)

                print("Preparing Blind-Spot Fusion Data......")
                noisy = data['noisy'].to(device)
                source = data['source'].to(device)
                mask = data['mask'].to(device)
                target = data['target'].to(device)

                # forward netG
                output = netG(source)

                output_grad = gradient(x=output, batch_size=output.shape[0])
                prior_grad = gradient(x=target, batch_size=output.shape[0])

                output_block_mean = mean_by_blocks(x=output, batch_size=output.shape[0])
                prior_block_mean = mean_by_blocks(x=target, batch_size=output.shape[0])

                mean_output_block_mean = output_block_mean.mean()
                mean_prior_block_mean = prior_block_mean.mean()

                zero_target = torch.from_numpy(np.zeros(output_grad.shape)).to(device)

                loss_G = lambda_exp * L2_Loss(output * (1 - mask), noisy * (1 - mask)) +\
                         lambda_grad * L2_Loss(output_grad, prior_grad) +\
                         lambda_TV * L1_Norm_Loss(output_grad, zero_target) +\
                         lambda_median * L2_Loss(mean_output_block_mean, mean_prior_block_mean)

                loss_G.backward()
                optimG.step()
                optimG.zero_grad()              # backward netG

                # get losses
                loss_G_train += [loss_G.item()]

                print('TRAIN: EPOCH %d: BATCH %04d/%04d: LOSS: %.4f'
                      % (epoch, batch, num_batch_train, np.mean(loss_G_train)))

                if should(num_freq_disp):
                    ## show output
                    noisy = transform_inv(noisy)
                    target = transform_inv(target)
                    output = transform_inv(output)

                    noisy = np.clip(noisy, 0, 1)
                    target = np.clip(target, 0, 1)
                    output = np.clip(output, 0, 1)
                    dif = np.clip(abs(target - noisy), 0, 1)

                    slice_size = int((noisy.shape[2])/2)

                    writer_train.add_images('noisy', noisy[:,:,slice_size,:,:], num_batch_train * (epoch - 1) + batch, dataformats='NCHW')
                    writer_train.add_images('output', output[:,:,slice_size,:,:], num_batch_train * (epoch - 1) + batch, dataformats='NCHW')
                    writer_train.add_images('target', target[:,:,slice_size,:,:], num_batch_train * (epoch - 1) + batch, dataformats='NCHW')

                torch.cuda.empty_cache()
            writer_train.add_scalar('loss_G', np.mean(loss_G_train), epoch)

            ## validation phase
            with torch.no_grad():
                netG.eval()

                loss_G_val = []

                for batch, data in enumerate(loader_val, 1):
                    def should(freq):
                        return freq > 0 and (batch % freq == 0 or batch == num_batch_val)

                    noisy = data['noisy'].to(device)
                    source = data['source'].to(device)
                    mask = data['mask'].to(device)
                    target = data['target'].to(device)

                    # forward netG
                    output = netG(source)

                    output_grad = gradient(x=output, batch_size=output.shape[0])
                    prior_grad = gradient(x=target, batch_size=output.shape[0])

                    output_block_mean = mean_by_blocks(x=output, batch_size=output.shape[0])
                    prior_block_mean = mean_by_blocks(x=target, batch_size=output.shape[0])

                    mean_output_block_mean = output_block_mean.mean()
                    mean_prior_block_mean = prior_block_mean.mean()

                    zero_target = torch.from_numpy(np.zeros(output_grad.shape)).to(device)

                    loss_G = lambda_exp * L2_Loss(output * (1 - mask), noisy * (1 - mask)) + \
                             lambda_grad * L2_Loss(output_grad, prior_grad) + \
                             lambda_TV * L1_Norm_Loss(output_grad, zero_target) + \
                             lambda_median * L2_Loss(mean_output_block_mean, mean_prior_block_mean)

                    # get losses
                    loss_G_val += [loss_G.item()]

                    print('VALID: EPOCH %d: BATCH %04d/%04d: LOSS: %.4f'
                          % (epoch, batch, num_batch_val, np.mean(loss_G_val)))

                    if should(num_freq_disp):
                        ## show output
                        noisy = transform_inv(noisy)
                        target = transform_inv(target)
                        output = transform_inv(output)

                        noisy = np.clip(noisy, 0, 1)
                        target = np.clip(target, 0, 1)
                        output = np.clip(output, 0, 1)
                        dif = np.clip(abs(target - noisy), 0, 1)

                        slice_size = int((noisy.shape[1]) / 2)

                        writer_train.add_images('noisy', noisy[:,:,slice_size,:,:], num_batch_val * (epoch - 1) + batch, dataformats='NCHW')
                        writer_train.add_images('output', output[:,:,slice_size,:,:], num_batch_val * (epoch - 1) + batch, dataformats='NCHW')
                        writer_train.add_images('target', target[:,:,slice_size,:,:], num_batch_val * (epoch - 1) + batch, dataformats='NCHW')

                writer_val.add_scalar('loss_G', np.mean(loss_G_val), epoch)

            ## save
            if (epoch % num_freq_save) == 0:
                self.save(dir_chck, netG, optimG, epoch)

        writer_train.close()
        writer_val.close()

    def test(self):
        mode = self.mode
        batch_size = self.batch_size
        device = self.device
        gpu_ids = self.gpu_ids

        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker

        tilesize = self.tilesize
        padding = self.padding

        norm = self.norm
        name_data = self.name_data
        outdir = self.dir_test_tomo

        ## setup dataset
        dir_chck = os.path.join(self.dir_checkpoint, name_data)
        # dir_result_test = os.path.join(self.dir_result, name_data, 'test')
        dir_data_test = os.path.join(self.dir_data, name_data, 'test')

        tomofile = os.listdir(dir_data_test)
        test_path = os.path.join(dir_data_test, tomofile[0])

        if len(tomofile) != 1:
            sys.exit(0)

        ## load test tomogram
        with open(test_path, 'rb') as f:
            content = f.read()
        tomo, header, _ = mrc.parse(content)
        tomo = tomo.astype(np.float32)
        name = os.path.basename(test_path)
        mu = tomo.mean()
        std = tomo.std()

        tomo = sampling(tomo, scale=2.0, mode='nearest')

        # header_filtered = header

        ## setup network
        netG = UNet3DEM(nch_in, nch_out, nch_ker, norm)
        netG = netG.cuda()
        netG = nn.DataParallel(netG)
        gaussian_layer = GaussianDenoise3d(sigma=0.5, scale=5)
        gaussian_layer = gaussian_layer.cuda()
        init_net(netG, init_type='kaiming', init_gain=0.02, gpu_ids=gpu_ids)
        denoised = np.zeros_like(tomo)
        # filtered_denoised = np.zeros_like(tomo)

        ## load from checkpoints
        netG, st_epoch = self.load(dir_chck, netG, mode=mode)
        patch_dataset_test = PatchBasedDataset(tomo=tomo, patch_size=tilesize, padding=padding)

        ## test phase
        with torch.no_grad():
            netG.eval()

            total = len(patch_dataset_test)
            print(str(total)+' patches are loaded for denoising...')
            count = 0
            loader_test = torch.utils.data.DataLoader(patch_dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

            for batch, data in enumerate(loader_test, 1):
                x = data['source'].to(device)
                index = data['pos']
                x = (x - mu)/std
                x = x.unsqueeze(1)  # batch x channel

                # denoise
                x = netG(x)
                x = gaussian_layer(x)
                x = x.squeeze(1).cpu().numpy()
                # x_filtered = x_filtered.squeeze(1).cpu().numpy()

                # stitch into denoised volume
                for b in range(len(x)):
                    i, j, k = index[b]
                    xb = x[b]
                    xb = xb * std + mu
                    patch = denoised[i:i+tilesize, j:j+tilesize, k:k+tilesize]
                    pz, py, px = patch.shape

                    xb = xb[padding:padding+pz, padding:padding+py, padding:padding+px]
                    denoised[i:i+tilesize, j:j+tilesize, k:k+tilesize] = xb

                    count += 1

        denoised = sampling(tomo=denoised, scale=0.5, mode='nearest')

        time_now = datetime.datetime.now()
        time_smpl = datetime.datetime.strftime(time_now, '%Y-%m-%d_%H:%M:%S')
        log_name = os.path.join(self.dir_checkpoint, time_smpl + '_' + name_data)
        # doc = open(log_name, 'w')

        # save the denoised tomogram
        basic_name = name
        name = time_smpl + '_' + basic_name
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outpath = outdir + os.sep + name

        # use the read header except for a few fields
        header = header._replace(mode=2)  # 32-bit real
        header = header._replace(amin=denoised.min())
        header = header._replace(amax=denoised.max())
        header = header._replace(amean=denoised.mean())


        with open(outpath, 'wb') as f:
            mrc.write(f, denoised, header=header)

