import torch
import torch.nn as nn
import torch.optim as optimizer
from torch.utils.data import DataLoader
from data_feed import DataFeed
import numpy as np
import time
from skimage import io, transform
import torchvision.transforms as trf
from build_net import RecNet


PATH_WEIGHTS = "/home/vinohithreddy/Self/ViWi/weights/model_3_pred.pth"
PATH_EMBED = "/home/vinohithreddy/Self/ViWi/weights/embed_3.txt"

options_dict = {
    'tag': 'Exp1_beam_seq_pred_no_images',
    'operation_mode': 'beams',

    # Data:
    'train_ratio': 1,
    'test_ratio': 1,
    'img_mean': (0.4905,0.4938,0.5285),
    'img_std':(0.05922,0.06468,0.06174),
    'val_data_file': '/home/vinohithreddy/Self/ViWi/dev_dataset_csv/dev_dataset_csv/val_set.csv',
    'results_file': 'five_beam_results_beam_only_2layeers.mat',

    # Net:
    'net_type':'gru',
    'cb_size': 128,  # Beam codebook size
    'out_seq': 3,  # Length of the predicted sequence
    'inp_seq': 8, # Length of inp beam and image sequence
    'embed_dim': 50,  # Dimension of the embedding space (same for images and beam indices)
    'hid_dim': 20,  # Dimension of the hidden state of the RNN
    'img_dim': [3, 160, 256],  # Dimensions of the input image
    'out_dim': 128,  # Dimensions of the softmax layers
    'num_rec_lay': 2,  # Depth of the recurrent network
    'drop_prob': 0.2,

    # Train param
    'gpu_idx': 0,
    'solver': 'Adam',
    'shf_per_epoch': True,
    'num_epochs': 50,
    'batch_size':5000,
    'val_batch_size':1000,
    'lr': 1e-3,
    'lr_sch': [200],
    'lr_drop_factor':0.1,
    'wd': 0,
    'display_freq': 50,
    'coll_cycle': 50,
    'val_freq': 100,
    'prog_plot': True,
    'fig_c': 0
}

resize = trf.Resize((options_dict['img_dim'][1],options_dict['img_dim'][2]))
normalize = trf.Normalize(mean=options_dict['img_mean'],
                          std=options_dict['img_std'])
transf = trf.Compose([
    trf.ToPILImage(),
    resize,
    trf.ToTensor(),
    normalize
])

val_feed = DataFeed(root_dir=options_dict['val_data_file'],
                     n=options_dict['inp_seq']+options_dict['out_seq'],
                     img_dim=tuple(options_dict['img_dim']),
                     transform=transf)
val_loader = DataLoader(val_feed,batch_size=1)
options_dict['test_size'] = val_feed.__len__()

weights_matrix = torch.from_numpy(np.loadtxt(PATH_EMBED))
weights_matrix = weights_matrix.type(torch.FloatTensor)
embed = nn.Embedding.from_pretrained(weights_matrix)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if options_dict['net_type'] == 'gru':
    net = RecNet(options_dict['embed_dim'],
                    options_dict['hid_dim'],
                    options_dict['out_dim'],
                    options_dict['out_seq'],
                    options_dict['num_rec_lay'],
                    options_dict['drop_prob'],
                    )

net.load_state_dict(torch.load(PATH_WEIGHTS, map_location=torch.device('cpu')))
num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Number of trainable parameters in the network : ", num_params)

net.eval()
batch_score = 0.
print("\tInput sequence\t\t        Prediction\t     Ground Truth")
with torch.no_grad():
    for v_batch, beam in enumerate(val_loader):
        init_beams = beam[:, :options_dict['inp_seq']].type(torch.LongTensor)
        inp_beams = embed(init_beams)
        # print(inp_beams)
        inp_beams = inp_beams.to(torch.device("cpu"))
        batch_size = beam.shape[0]

        targ = beam[:,options_dict['inp_seq']:options_dict['inp_seq']+options_dict['out_seq']]\
               .type(torch.LongTensor)
        targ = targ.view(batch_size,options_dict['out_seq'])
        targ = targ.to(torch.device("cpu"))
        h_val = net.initHidden(beam.shape[0]).to(torch.device("cpu"))
        out, h_val = net.forward(inp_beams, h_val)
        pred_beams = torch.argmax(out, dim=2)
        print(" ", init_beams[0].numpy(), "   ", pred_beams[0].numpy(), "\t    ", beam[0,options_dict['inp_seq']:options_dict['inp_seq']+options_dict['out_seq']].numpy())
        batch_score += torch.sum( torch.prod( pred_beams == targ, dim=1, dtype=torch.float ) )
print("Number of correct predictions : ", batch_score.numpy())