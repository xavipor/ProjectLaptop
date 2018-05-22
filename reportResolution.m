% This script is to save each volume as a .mat file to input to cnn
% The saved index is ordered by the id
clear;
addpath('./NIfTI_20140122/')

mode = 'test';
img_data_path = '/media/haocheng/DATA_1T/IMAGES/cmb-3dcnn-data/raw_data_upsampled/';
img_data_path_1 = '/media/haocheng/DATA_1T/IMAGES/cmb-3dcnn-data/raw_data/';

files = dir(img_data_path);
files(1:2)=[];
num = length(files);

for jj = 1:1:num
% for jj = 1
    name = files(jj).name;
    fprintf('Loading No.%d %s subject %s (total %d).\n', jj, mode, name, num);
    
    o = load_untouch_nii([img_data_path_1 files(jj).name]);
    fprintf('Original size: %d x %d x %d\n', size(o.img));
    
    up = load_untouch_nii([img_data_path files(jj).name]);
%     nii.hdr.dime.pixdim(2:4)
    fprintf('Upsampled size: %d x %d x %d\n', size(up.img));
    
    
    
end





    
    