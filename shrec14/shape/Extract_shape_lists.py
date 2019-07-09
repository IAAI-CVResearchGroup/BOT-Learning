import pdb
import csv
import glob
from tqdm import tqdm

shape_label_dir = '/data/shrec14/shape_labels.csv'
list_file = open('/home/sri-dev01/CVRP_New/shrec14/shape/shape_sh.txt', 'w')

shape_label_f = open(shape_label_dir, 'r')
with open(shape_label_dir,'r') as sl_csv:
    sl_reader = csv.reader(sl_csv)
    sl_list = list(sl_reader)

model_name = []
for ind1 in range(len(sl_list)):
    model_name.append(sl_list[ind1][0])

files = glob.glob('/data/shrec14/Shape_Features_20Views/*')
files.sort()
for ind in tqdm(range(len(files))):
    filename = files[ind][-14:-8]
    label_loc = model_name.index(filename)
    label = int(sl_list[label_loc][1])
    list_file.write('{} {}\n'.format(files[ind], label))
#pdb.set_trace()