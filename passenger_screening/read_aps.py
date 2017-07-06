import numpy as np
import glob
import matplotlib
matplotlib.use("TkAgg")
matplotlib.rc('animation', html='html5')
from matplotlib import animation
from matplotlib import pyplot as plt
import os

HEADER_SIZE = 512

def count_aps(src):
  with open(src, 'r+b') as fd:
    while True:
      buf = np.fromfile(fd, dtype='S1', count=20)
      print(len(buf), buf)
      if len(buf) == 0:
        break
      data.append(buf)

def read_header(src):
  header = {}
  with open(src, 'r+b') as fd:
    header['filename'] = b''.join(np.fromfile(fd, dtype='S1', count=20))
    header['parent_filename'] = b''.join(np.fromfile(fd, dtype='S1', count=20))
    header['comments1'] = b''.join(np.fromfile(fd, dtype='S1', count=80))
    header['comments2'] = b''.join(np.fromfile(fd, dtype='S1', count=80))
    header['energy_type'] = np.fromfile(fd, dtype=np.int16, count=1)
    header['config_type'] = np.fromfile(fd, dtype=np.int16, count=1)
    header['file_type'] = np.fromfile(fd, dtype=np.int16, count=1)
    header['trans_type'] = np.fromfile(fd, dtype=np.int16, count=1)
    header['scan_type'] = np.fromfile(fd, dtype=np.int16, count=1)
    header['data_type'] = np.fromfile(fd, dtype=np.int16, count=1)
    header['date_modified'] = b''.join(np.fromfile(fd, dtype='S1', count=16))
    header['frequency'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['mat_velocity'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['num_pts'] = np.fromfile(fd, dtype=np.int32, count=1)
    header['num_polarization_channels'] = np.fromfile(fd, dtype=np.int16, count=1)
    header['spare00'] = np.fromfile(fd, dtype=np.int16, count=1)
    header['adc_min_voltage'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['adc_max_voltage'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['band_width'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['spare01'] = np.fromfile(fd, dtype=np.int16, count=5)
    header['polarization_type'] = np.fromfile(fd, dtype=np.int16, count=4)
    header['record_header_size'] = np.fromfile(fd, dtype=np.int16, count=1)
    header['word_type'] = np.fromfile(fd, dtype=np.int16, count=1)
    header['word_precision'] = np.fromfile(fd, dtype=np.int16, count=1)
    header['min_data_value'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['max_data_value'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['avg_data_value'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['data_scale_factor'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['data_units'] = np.fromfile(fd, dtype=np.int16, count=1)
    header['surf_removal'] = np.fromfile(fd, dtype=np.uint16, count=1)
    header['edge_weighting'] = np.fromfile(fd, dtype=np.uint16, count=1)
    header['x_units'] = np.fromfile(fd, dtype=np.uint16, count=1)
    header['y_units'] = np.fromfile(fd, dtype=np.uint16, count=1)
    header['z_units'] = np.fromfile(fd, dtype=np.uint16, count=1)
    header['t_units'] = np.fromfile(fd, dtype=np.uint16, count=1)
    header['spare02'] = np.fromfile(fd, dtype=np.int16, count=1)
    header['x_return_speed'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['y_return_speed'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['z_return_speed'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['scan_orientation'] = np.fromfile(fd, dtype=np.int16, count=1)
    header['scan_direction'] = np.fromfile(fd, dtype=np.int16, count=1)
    header['data_storage_order'] = np.fromfile(fd, dtype=np.int16, count=1)
    header['scanner_type'] = np.fromfile(fd, dtype=np.int16, count=1)
    header['x_inc'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['y_inc'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['z_inc'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['t_inc'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['num_x_pts'] = np.fromfile(fd, dtype=np.int32, count=1)
    header['num_y_pts'] = np.fromfile(fd, dtype=np.int32, count=1)
    header['num_z_pts'] = np.fromfile(fd, dtype=np.int32, count=1)
    header['num_t_pts'] = np.fromfile(fd, dtype=np.int32, count=1)
    header['x_speed'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['y_speed'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['z_speed'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['x_acc'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['y_acc'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['z_acc'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['x_motor_res'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['y_motor_res'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['z_motor_res'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['x_encoder_res'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['y_encoder_res'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['z_encoder_res'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['date_processed'] = b''.join(np.fromfile(fd, dtype='S1', count=8))
    header['time_processed'] = b''.join(np.fromfile(fd, dtype='S1', count=8))
    header['depth_recon'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['x_max_travel'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['y_max_travel'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['elevation_offset_angle'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['roll_offset_angle'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['z_max_travel'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['azimuth_offset_angle'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['adc_type'] = np.fromfile(fd, dtype=np.int16, count=1)
    header['spare06'] = np.fromfile(fd, dtype=np.int16, count=1)
    header['scanner_radius'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['x_offset'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['y_offset'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['z_offset'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['t_delay'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['range_gate_start'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['range_gate_end'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['ahis_software_version'] = np.fromfile(fd, dtype=np.float32, count=1)
    header['spare_end'] = np.fromfile(fd, dtype=np.float32, count=10)
  return header

def read_data(fsrc, header):
  x, y, t = int(header.get('num_x_pts')),\
              int(header.get('num_y_pts')),\
              int(header.get('num_t_pts'))
  with open(fsrc, 'rb') as fd:
    fd.seek(HEADER_SIZE)
    buf = np.fromfile(fd, dtype=np.uint16, count=x*y*t)
  #buf = buf.astype(np.float32) * header.get('data_scale_factor')
  buf = np.reshape(buf, (x, y, t), order='F')
#  plot_image(buf)
#  show_frame(buf)
  return buf, x*y*t

def show_frame(data):
  plt.ion()
  fig = plt.figure(figsize=(16, 16), frameon=True, edgecolor='black')
  rows, cols = 4, 4
  fig.show()
  for i in range(data.shape[2]):
    ax = fig.add_subplot(data.shape[2], 1, 1)#i//cols+1, i%rows+1))
    ax.imshow(data[:,:,i])

def plot_image(data):
  fig = plt.figure(figsize=(8, 8), facecolor='darkgray', edgecolor='black')
  ax = fig.add_subplot(111)
  def animate(i):
    im = ax.imshow(np.flipud(data[:,:,i].transpose()), cmap = 'viridis')
    return [im]
  return animation.FuncAnimation(fig, animate, frames=range(0, data.shape[2]), interval=200, blit=True)

def try_read():
  total = 0
  data_root = os.path.join(os.path.dirname(__file__), '../data/passenger_screening/stage1_aps')

  for src in glob.glob(data_root+'/*.aps'):
    header = read_header(src)
    data, sz = read_data(src, header)
    break

def init_plot():
  plt.ion()
  plt.show()


