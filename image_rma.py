import torch
import argparse
from scipy.io import loadmat,savemat
from models import *
from datasets import *
from utils import *
import tqdm
import time
## 成像输入： 1、回波矩阵输入路径 2、相位补偿矩阵 3、成像算法 4、成像结果保存路径

parser= argparse.ArgumentParser()
parser.add_argument("--echo_dir", type=str, default='./puca/mat_save/0.4', help="echo_dir")
parser.add_argument("--item_name", type=str, default="echo_mnist_1000.mat", help="name of the echo")
parser.add_argument("--echo_name", type=str, default="data", help="name of the echo")
parser.add_argument("--save_dir", type=str, default='./puca/image_rma', help='save directory')
parser.add_argument("--phase_imag_dir", type=str, default='../dataset/phaseImag.mat', help='save directory')
parser.add_argument("--phase_mese_dir", type=str, default='../dataset/phaseMese.mat', help='model name')
parser.add_argument("--test_num", type=int, default=1, help='test number')
parser.add_argument("--save_name", type=str, default='image_result.mat', help='model name')

opt = parser.parse_args()
print(opt)
get_sparse_echo = loadmat(os.path.join(opt.echo_dir,opt.item_name))['ori_data'][:]
get_complete_echo = loadmat(os.path.join(opt.echo_dir,opt.item_name))['data'][:]

get_phase_imag = h5py.File(opt.phase_imag_dir)['phaseImag'][:].transpose()
get_phase_mese = h5py.File(opt.phase_mese_dir)['phaseMese'][:].transpose()


get_sparse_echo = torch.from_numpy(get_sparse_echo).cuda()
get_complete_echo = torch.from_numpy(get_complete_echo).cuda()  
get_phase_imag = torch.from_numpy(get_phase_imag).cuda()
get_phase_mese = torch.from_numpy(get_phase_mese).cuda()

if opt.test_num==1:
    get_phase_imag = get_phase_imag.unsqueeze(0)
    get_phase_mese = get_phase_mese.unsqueeze(0)

image_result1 = []
image_result2 = []
for i in tqdm.tqdm(range(opt.test_num)):
    
    # image_result1.append(RmaImaging_torch(get_sparse_echo[i],get_phase_imag[i,...]))
    # image_result2.append(RmaImaging_torch(get_complete_echo[i],get_phase_imag[i,...]))
    image_result1.append(RmaImaging_torch(get_sparse_echo[i],get_phase_imag[i,...]))
    image_result2.append(RmaImaging_torch(get_complete_echo[i],get_phase_imag[i,...]))
# print('Time:',time2-time1)
image_result1 = torch.cat(image_result1,dim=0)
image_result2 = torch.cat(image_result2,dim=0)

if not os.path.exists(opt.save_dir):
    os.makedirs(opt.save_dir)

savemat(os.path.join(opt.save_dir,opt.save_name),{'sparse_echo':get_sparse_echo.cpu().numpy(),'complete_echo':get_complete_echo.cpu().numpy(),'sparse_imag':image_result1.cpu().numpy(),'complete_imag':image_result2.cpu().numpy(),'phase_imag':get_phase_imag.cpu().numpy(),'phase_mese':get_phase_mese.cpu().numpy()})
