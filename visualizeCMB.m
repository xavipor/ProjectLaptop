function visualizeCMB()

% img_data_path = '/media/haocheng/2D1E-18F9/IMAGES/GoDARTS_Imaging_MR/subset_resampled/';
img_data_path = '/media/haocheng/DATA_1T/IMAGES/cmb-3dcnn-data/raw_data/';
files = dir(img_data_path);
files(1:2)=[];
num = length(files);

load('/media/haocheng/DATA_1T/CODE/cmb-3dcnn-code-v1.0/result1/detection.mat')
% volumeSize = [512,512,144];

% gt_mat_path = '/media/haocheng/DATA_1T/CODE/cmb-3dcnn-code-v1.0/code/gt_mat/';
gt_mat_path = '/media/haocheng/DATA_1T/IMAGES/cmb-3dcnn-data/ground_truth/';

for jj = 9
% for jj = 1
    name = files(jj).name;
     
    fprintf('Visual No.%d subject %s (total %d).\n', jj, name, num);
    nii = load_untouch_nii([img_data_path files(jj).name]);
    V = nii.img;
%     V = resizeVolume(V, volumeSize);
    
    dcen = cmb{jj};
    
%     load([gt_mat_path num2str(jj) '.mat']);
    load([gt_mat_path name(1:2), '.mat']);
    
    cen1 = cen(5:9,:);
    cen1(:,3) = 79;
    
    figure, imshow(V(:,:,79), []);
    hold on
    for i = 1:1:size(cen1,1)
        drawBox(cen1(i,2), cen1(i,1), 11, 'blue')
    end
    
    dcen1 = dcen(5:8,:);
    dcen1(:,3) = 79;
    
    figure, imshow(V(:,:,79), []);
    hold on
    for i = 1:1:size(dcen1,1)
        drawBox(dcen1(i,2), dcen1(i,1), 11, 'red')
    end
    
       
end

end


function drawBox(x, y, s, col)

xs = x-(s-1)/2;
ys = y-(s-1)/2;

rectangle('Position',[xs,ys,s,s], 'EdgeColor', col, 'LineWidth', 0.8);

end


    