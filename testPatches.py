import numpy as np
import h5py

input_sizes = [452,512,144] #[in_height,in_width,in_time]
output_sizes = [219,249,68]       
clip_rate = [3,3,2]
whole_volume_path = '../result_dundee_10/mat_data/'
mode = 'test'
M_layer = 1
patch_size = [16,16,10]

in_height, in_width, in_time = input_sizes

dim0_score_start_pos = []
dim0_score_end_pos = []
dim0_start_pos = []
dim0_end_pos = []  
for part in range(clip_rate[0]):
    dim0_score_start_pos.append(1+part*output_sizes[0]/clip_rate[0])
    dim0_score_end_pos.append((part+1)*output_sizes[0]/clip_rate[0])
    dim0_start_pos.append(2*M_layer*(1+part*output_sizes[0]/clip_rate[0]-1)+1)
    dim0_end_pos.append(2*M_layer*((part+1)*output_sizes[0]/clip_rate[0]-1)+patch_size[0])   
dim0_pos = zip(dim0_start_pos,dim0_end_pos)
dim0_score_pos = zip(dim0_score_start_pos,dim0_score_end_pos)
print('dim0_pos')
print(dim0_pos)
print('dim0_score_pos')
print(dim0_score_pos)

dim1_score_start_pos = []
dim1_score_end_pos = []
dim1_start_pos = []
dim1_end_pos = []
for part in range(clip_rate[1]):
    dim1_score_start_pos.append(1+part*output_sizes[1]/clip_rate[1])
    dim1_score_end_pos.append((part+1)*output_sizes[1]/clip_rate[1])
    dim1_start_pos.append(2*M_layer*(1+part*output_sizes[1]/clip_rate[1]-1)+1)
    dim1_end_pos.append(2*M_layer*((part+1)*output_sizes[1]/clip_rate[1]-1)+patch_size[1])   
dim1_pos = zip(dim1_start_pos,dim1_end_pos)
dim1_score_pos = zip(dim1_score_start_pos,dim1_score_end_pos)
print('dim1_pos')
print(dim1_pos)
print('dim1_score_pos')
print(dim1_score_pos)

dim2_score_start_pos = []
dim2_score_end_pos = []
dim2_start_pos = []
dim2_end_pos = []
for part in range(clip_rate[2]):
    dim2_score_start_pos.append(1+part*output_sizes[2]/clip_rate[2])
    dim2_score_end_pos.append((part+1)*output_sizes[2]/clip_rate[2])
    dim2_start_pos.append(2*M_layer*(1+part*output_sizes[2]/clip_rate[2]-1)+1)
    dim2_end_pos.append(2*M_layer*((part+1)*output_sizes[2]/clip_rate[2]-1)+patch_size[2])   
dim2_pos = zip(dim2_start_pos,dim2_end_pos)
dim2_score_pos = zip(dim2_score_start_pos,dim2_score_end_pos)
print('dim2_pos')
print(dim2_pos)
print('dim2_score_pos')
print(dim2_score_pos)

score_mask = np.zeros((2,output_sizes[0],output_sizes[1],output_sizes[2]))
print('score_mask')
print(score_mask.shape)

data_path = whole_volume_path + str(1) + '_' + mode + '.mat'
data_set = np.transpose(np.array(h5py.File(data_path)['data']))      
data_set = data_set - np.mean(data_set)
data_set = data_set.reshape((data_set.shape[0],in_time,1,in_height,in_width))
print(data_set.shape)

for dim2 in range(clip_rate[2]):
    for dim1 in range(clip_rate[1]):
            for dim0 in range(clip_rate[0]):
                #sys.stdout.write('.')
                print('dim2:')
                print(dim2_pos[dim2][0]-1)
                print(dim2_pos[dim2][1])

                print('dim0')
                print(dim0_pos[dim0][0]-1)
                print(dim0_pos[dim0][1])

                print('dim1')
                print(dim1_pos[dim1][0]-1)
                print(dim1_pos[dim1][1])

                smaller_data = data_set[:,dim2_pos[dim2][0]-1:dim2_pos[dim2][1],:,dim0_pos[dim0][0]-1:dim0_pos[dim0][1],dim1_pos[dim1][0]-1:dim1_pos[dim1][1]]
                print(smaller_data.shape)