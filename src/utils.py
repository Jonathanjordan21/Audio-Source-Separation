def calculate_conv_size(shape, kernel_size, stride=1, padding=0, n_times=1):
  if(type(kernel_size)==int):
    kernel_size = (kernel_size, kernel_size)
  if(type(stride)==int):
    stride = (stride, stride)
  shape = list(shape)

  for _ in range(n_times):
    shape[0] = int((shape[0] + 2*padding - kernel_size[0])/stride[0]) +1
    shape[1] = int((shape[1] + 2*padding - kernel_size[1])/stride[1]) +1
  return shape
  