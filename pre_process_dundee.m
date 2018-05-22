% This script is to save each volume as a .mat file to input to cnn
% The saved index is ordered by the id
clear;
addpath('./NIfTI_20140122/')

mode = 'test';
img_data_path = '/home/xavipor/Documentos/Microbleeds/GoDARTS/Images/';
save_datasets_path = '/home/xavipor/Imágenes/DundeePreprocessing';
if ~exist(save_datasets_path)
    mkdir(save_datasets_path);
end

files = dir(img_data_path);
files(1:2)=[];
num = length(files);

volumeSize = [512,512,144];

fprintf('Extracting %s dataset ... \n',mode);
for jj = 1:1:num
% for jj = 1
    name = files(jj).name;
    fprintf('Loading No.%d %s subject %s (total %d).\n', jj, mode, name, num);
    nii = load_untouch_nii([img_data_path files(jj).name]);
    V = nii.img;
    data_volume = resizeVolume(V, volumeSize);
    if size(data_volume) ~= volumeSize
        error('something wrong')
    end
%    for i = 1:5:144
%          figure, imshow(data_volume(:,:,i), [])
%    end
%    data_volume = nii.img;
%    close all
    
    data_volume = (data_volume - min(data_volume(:)))./(max(data_volume(:)) - min(data_volume(:)));
    
    data = single(reshape(data_volume,[1 prod(size(data_volume))]));
    save([save_datasets_path num2str(jj) '_' mode '.mat'],'data','-v7.3')  
    clear nii data data_volume
end





    
    