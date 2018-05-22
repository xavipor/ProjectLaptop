% This script is to save each volume as a .mat file to input to cnn
% The saved index is ordered by the id
clear;clc;
addpath('./NIfTI_20140122/')

mode = 'test';
img_data_path = '~/Documentos/Microbleeds/Paper/raw_data/';
save_datasets_path = '~/Imágenes/ImagesPreprocessing/';
if ~exist(save_datasets_path)
    mkdir(save_datasets_path);
end

x = 512;
y = 512;
z = 150;

files = dir(img_data_path);
files(1:2)=[]%To delete the first to elements that are junk
num = length(files);

fprintf('Extracting %s dataset ... \n',mode);
for jj = 1:num
    data_volume = zeros(x,y,z);
    fprintf('Loading No.%d %s subject (total %d).\n', jj,mode,num);
    nii = load_untouch_nii([img_data_path files(jj).name]);
    [xrange yrange zrange] = size(nii.img);
    data_volume(1:xrange,1:yrange,1:zrange) = nii.img(:,:,:);
    data_volume = data_volume(1:end,1:end,1:end-2);
    data = single(reshape(data_volume,[1 prod(size(data_volume))]));
    save([save_datasets_path num2str(jj) '_' mode '.mat'],'data','-v7.3');  
    clear nii data data_volume
end

exit




    
    