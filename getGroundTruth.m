function getGroundTruth()

img_path = '/media/haocheng/2D1E-18F9/IMAGES/GoDARTS_Imaging_MR/subset_resampled/';
gt_path = '/media/haocheng/2D1E-18F9/IMAGES/GoDARTS_Imaging_MR/subset_resampled_gt_refined/';

gt_mat_path = '/media/haocheng/DATA_1T/CODE/cmb-3dcnn-code-v1.0/code/gt_mat/';
if ~exist(gt_mat_path)
    mkdir(gt_mat_path);
end

% load('/media/haocheng/DATA_1T/CODE/cmb-3dcnn-code-v1.0/result_dundee_10/detection.mat');

files = dir(gt_path);
files(1:2)=[];
num = length(files);

volumeSize = [512,512,144];

for jj = 1:1:num
% for jj = 1
    name = files(jj).name;
    fprintf('Loading No.%d subject %s (total %d).\n', jj, name, num);
    
    % load image
%     nii = load_untouch_nii([img_path files(jj).name]);
%     V = nii.img;
%     V = resizeVolume(V, volumeSize);
    
    % load gt
    G = load_untouch_nii([gt_path files(jj).name]);
    GT = G.img;
    GT = resizeVolume(GT, volumeSize);
    
    [L,NUM] = bwlabeln(GT,18);
    s = regionprops(L, 'centroid');
    cen = cat(1, s.Centroid);
    cen = floor(cen);
    
%     dcen = cmb{1,1};
%     for i = 1:1:size(cen,1)
%         figure, imshow(V(:,:,cen(i,3)), [])
%         hold on, plot(cen(i,1), cen(i,2), '*r')
%     end
    cen(:,[1,2]) = cen(:,[2,1]);
    gt_num = NUM;
    save([gt_mat_path num2str(jj) '.mat'], 'cen', 'gt_num');
    
end


end
    
    