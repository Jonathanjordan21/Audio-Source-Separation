import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import calculate_conv_size

# Constants According to the Paper
F_c = 24
f_u = 5
f_d = 15
L = 12

class WaveUNet(nn.Module):
  
  def __init__(self, num_channel=1, num_sources=2, layers=12, shape=(1,441000)):
    super().__init__()
    self.down = nn.ModuleList()
    self.up = nn.ModuleList()

    curr_shape = shape
    curr_channel = num_channel
    shapes = []

    for i in range(1,layers+1):
      self.down = self.down.append(self.down_block(curr_channel, F_c*i, f_d))
      curr_channel = F_c*i

      curr_shape = list(calculate_conv_size(curr_shape, f_d))
      curr_shape[0] = curr_channel
      shapes.append(tuple(curr_shape))
      curr_shape[-1] = int((curr_shape[-1]+1)/2)
      
    self.bottleneck = nn.Conv1d(curr_channel, F_c*(layers+1), 1)
    curr_channel = F_c*(layers+1)

    curr_shape = list(calculate_conv_size(curr_shape, 1))
    curr_shape[0] = curr_channel
    shapes.append(curr_shape)

    shapes.reverse()
    for i,shape in zip(reversed(range(1,layers+1)), shapes[1:]):
      self.up = self.up.append(self.up_block(curr_channel+shape[0], F_c*i, f_u))
      curr_channel = F_c*i

    self.last_conv = nn.Conv1d(curr_channel+num_channel, num_channel*num_sources, 1)
    self.last_tanh = nn.Tanh()
    self.float()

  def down_block(self, in_channel, filters = 24,kernel_size=15):
    return nn.Sequential(
        nn.Conv1d(in_channel, filters, kernel_size, padding=(kernel_size-1)//2),
        nn.LeakyReLU()
    )
  def up_block(self, in_channel, filters = 24,kernel_size=5):
    return nn.Sequential(
        nn.ConvTranspose1d(in_channel, filters, kernel_size, padding=(kernel_size-1)//2),
        nn.LeakyReLU(),
        
    )
  
  def forward(self, x):
    down_out = [x]
    for down in self.down:
      x = down(x)
      down_out.append(x)
      # Decimation
      x = x[:,:,::2]

    x = self.bottleneck(x)
    down_out.reverse()

    # upsample frequency
    for up,out in zip(self.up, down_out[:-1]):
      inter = F.interpolate(x, mode='linear', scale_factor=2)
      x=torch.cat((inter[:, :, :out.shape[-1]], out),1)
      x = up(x)
    out = down_out[-1]
    x = torch.cat((x,out[:,:,:x.shape[-1]]),1)
    return self.last_tanh(self.last_conv(x))