function get_datasets_final_pred_dundee(result_path,xdim,ydim,zdim)
    
    % This script is to extract cubic candidates to wrap into the final
    % prediction model
    img_data_path = '/media/haocheng/2D1E-18F9/IMAGES/GoDARTS_Imaging_MR/subset_resampled/';
    files = dir(img_data_path);
    files(1:2) = [];
    score_map_cand_path = [result_path 'score_map_cands/'];
    datasets_path = [result_path 'test_set_cand/']; 
    if ~exist(datasets_path)
        mkdir(datasets_path);
    end   
    % Start to extract the test candidates cubics %%%%%%%%%%%%%%%%%%%%%%
    fprintf('Extracting testing candidates ...\n');
    tic;

    volumeSize = [512,512,144];
    
    for jj = 1:length(files)
%     for jj = 1
        name = files(jj).name;
        counter = 0;
        test_set_x = zeros([2000 xdim*ydim*zdim]);   
        fprintf('Loading No.%d subject %s(total %d).\n', jj, name, length(files));
        nii = load_untouch_nii([img_data_path name]);
        V = nii.img;
        V = resizeVolume(V, volumeSize);
        if size(V) ~= volumeSize
            error('something wrong')
        end
        V = (V - min(V(:)))./(max(V(:)) - min(V(:)));
        
        load([score_map_cand_path num2str(jj) '_cand.mat']);
        [b_x b_y b_z] = size(V);
        for i = 1:size(center,1)
            counter = counter + 1;
            pp1 = center(i,1);
            pp2 = center(i,2);
            pp3 = center(i,3);            
            if (pp1<xdim/2 || pp1>b_x-xdim/2 || pp2<ydim/2 || pp2>b_y-ydim/2 || pp3<zdim/2 || pp3>b_z-zdim/2)
                patch = zeros(xdim,ydim,zdim);
            else
                patch = V(pp1-xdim/2+1:pp1+xdim/2,pp2-ydim/2+1:pp2+ydim/2,pp3-zdim/2+1:pp3+zdim/2);
            end
            test_set_x(counter,:) = reshape(patch, xdim*ydim*zdim, []);       
            clear patch
        end
        test_set_x(counter+1:end,:)=[];
        save([datasets_path num2str(jj) '_patches.mat'],'test_set_x','-v7.3');
    end
    toc;


end









