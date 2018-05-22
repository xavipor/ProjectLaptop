function visualizeCMB_Tayside()

img_data_path = '/media/haocheng/2D1E-18F9/IMAGES/GoDARTS_Imaging_MR/subset_resampled/';
files = dir(img_data_path);
files(1:2)=[];
num = length(files);

load('/media/haocheng/DATA_1T/CODE/cmb-3dcnn-code-v1.0/result_dundee_10/detection.mat')
volumeSize = [512,512,144];

gt_mat_path = '/media/haocheng/DATA_1T/CODE/cmb-3dcnn-code-v1.0/code/gt_mat/';

for jj = 2
% for jj = 1
    name = files(jj).name;
     
    fprintf('Visual No.%d subject %s (total %d).\n', jj, name, num);
    nii = load_untouch_nii([img_data_path files(jj).name]);
    V = nii.img;
    V = resizeVolume(V, volumeSize);
    
    dcen = cmb{jj};
    
    load([gt_mat_path num2str(jj) '.mat']);
    
%     for i = 1:1:size(cen,1)
%         figure, imshow(V(:,:,cen(i,3)), []), title(num2str(cen(i,3)))
%         drawBox(cen(i,2), cen(i,1), 11, 'blue')
%     end
    
%     for i = 1:1:size(dcen,1)
%         figure, imshow(V(:,:,dcen(i,3)), []), title(num2str(dcen(i,3)))
%         drawBox(dcen(i,2), dcen(i,1), 11, 'red')
%     end

    %% jj=2
    cen1 = cen(12:14,:);
    cen1(:,3) = 77;
    figure, imshow(V(:,:,77), []);
    hold on
    for i = 1:1:size(cen1,1)
        drawBox(cen1(i,2), cen1(i,1), 11, 'blue')
    end
%     
    dcen1 = dcen(37:41,:);
    dcen1(:,3) = 77;
    
    figure, imshow(V(:,:,77), []);
    hold on
    for i = 1:1:size(dcen1,1)
        drawBox(dcen1(i,2), dcen1(i,1), 11, 'red')
    end
%     
    
    %% jj = 3
%     cen1 = cen(1:3,:);
%     cen1(:,3) = 62;
% %     
%     figure, imshow(V(:,:,62), []);
%     hold on
%     for i = 1:1:size(cen1,1)
%         drawBox(cen1(i,2), cen1(i,1), 11, 'blue')
%     end
% %     
%     dcen1 = dcen(9:11,:);
%     dcen1(:,3) = 62;
%     
%     figure, imshow(V(:,:,62), []);
%     hold on
%     for i = 1:1:size(dcen1,1)
%         drawBox(dcen1(i,2), dcen1(i,1), 11, 'red')
%     end
    
       
end

end


function drawBox(x, y, s, col)

xs = x-(s-1)/2;
ys = y-(s-1)/2;

rectangle('Position',[xs,ys,s,s], 'EdgeColor', col, 'LineWidth', 0.8);

end


    