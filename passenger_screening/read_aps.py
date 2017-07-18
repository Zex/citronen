import matplotlib
matplotlib.use("TkAgg")
matplotlib.rc('animation', html='html5')
import numpy as np
import glob
from matplotlib import animation
from matplotlib import pyplot as plt
import os, sys
from pandas import read_csv
from os.path import isfile, basename

HEADER_SIZE = 512

def count_aps(src):
  with open(src, 'r+b') as fd:
    while True:
      buf = np.fromfile(fd, dtype='S1', count=20)
      print(len(buf), buf)
      if len(buf) == 0:
        break
      data.append(buf)

def load_labels(label_path):
  # Columns:
  #  - Id,Probability
  if not isfile(label_path):
    raise FileNotFoundError(label_path)
  labels = read_csv(label_path, header=0)  
  return labels

def get_label(labels, iid, i):
  lid = '{}_Zone{}'.format(iid, str(i+1))
  #y.extend(labels[labels['Id'] == lid]['Probability'].values)
  y = labels[labels['Id'] == lid]['Probability'].values
  return y

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

def read_data(src, header):
  x, y, t = int(header.get('num_x_pts')),\
              int(header.get('num_y_pts')),\
              int(header.get('num_t_pts'))
  with open(src, 'rb') as fd:
    fd.seek(HEADER_SIZE)
    buf = np.fromfile(fd, dtype=np.uint16, count=x*y*t)
  t = 64
  #buf = buf.astype(np.float32) * header.get('data_scale_factor')
  buf = np.reshape(buf, (x, y, t), order='F')
  if __name__ == '__main__':
    iid = basename(src).split('.')[0]
    show_frame(buf, iid)
  return buf, x*y*t

label_path = '../data/passenger_screening/stage1_labels.csv'
labels = load_labels(label_path)

def show_frame(data, iid):
  global fig, ax
  for i in range(data.shape[2]):
    y = get_label(labels, iid, i)
    if y.squeeze() != 1:
      continue
    ax.imshow(data[:,:,i])
    fig.canvas.draw()

def plot_image(data):
  fig = plt.figure(figsize=(4, 4), facecolor='darkgray', edgecolor='black')
  ax = fig.add_subplot(111)
  def animate(i):
    im = ax.imshow(np.flipud(data[:,:,i].transpose()), cmap = 'viridis')
    return [im]
  return animation.FuncAnimation(fig, animate, frames=range(0, data.shape[2]), interval=200, blit=True)

def show_header(header):
  for k, v in header.items():
    print('{}: {}'.format(k, v))

def try_read():
  total = 0
  #data_root = os.path.join(os.path.dirname(__file__), '../data/passenger_screening/stage1_aps')
  data_root = os.path.join(os.path.dirname(__file__), '../data/passenger_screening/stage1_a3daps')

  for src in glob.glob(data_root+'/*'):
    try:
      header = read_header(src)
      #show_header(header)
      data, sz = read_data(src, header)
    except KeyboardInterrupt:
      print('terminated')
      sys.exit()
    except Exception as ex:
      print('try read failed', ex)

def init_plot():
  plt.ion()
  plt.show()

if __name__ == '__main__':
  global fig, ax
  init_plot()
  fig = plt.figure(figsize=(4, 4), facecolor='black', edgecolor='black')
  ax = fig.add_subplot(111)
  plt.ion()
  fig.show()
  try_read()
