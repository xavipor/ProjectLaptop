%% Generate patches without microbleeds
patch_size=[16 16 10];


resampled_img_files= dir(resampled_img_path);
resampled_gt_files = dir(resampled_GT_path);

resampled_img_files(1:2)=[];%To delete the first to elements that are junk
resampled_gt_files(1:2)=[];%To delete the first to elements that are junk

auxPatchSize=[16 16 10]/2;
center_list=[0,0,0];
l1=length(resampled_img_files);
counter2=0;
for jj = 1:l1
    

    
    center_list=[0,0,0];
    %fprintf('Loading No.%d %s subject (total %d).\n', jj,mode,num);
    nii_resampled_img = load_untouch_nii([resampled_img_path resampled_img_files(jj).name]);
    nii_resampled_gt = load_untouch_nii([resampled_GT_path resampled_gt_files(jj).name]);

    
    resampled_img = nii_resampled_img.img;
    resampled_gt = nii_resampled_gt.img ;
    
    imgSize = size(resampled_img);
    %Gt and Image hace the same size..
    

    
    counter=1;
   
    while counter < 500
        rndZ = floor(imgSize(3)*0.8 + (imgSize(3)-imgSize(3)).*rand());
        
        if rndZ > 0.6*imgSize(3) 
            rndX = floor(imgSize(1)*0.3 + (imgSize(1)*0.7-imgSize(1)*0.3).*rand());
            rndY = floor(imgSize(2)*0.3 + (imgSize(2)*0.7-imgSize(2)*0.3).*rand());
        elseif(rndZ <= 0.6*imgSize(3) || rndZ > 0.4*imgSize(3))
            rndX = floor(imgSize(1)*0.65 + (imgSize(1)*0.7-imgSize(1)*0.3).*rand());
            rndY = floor(imgSize(2)*0.55 + (imgSize(2)*0.7-imgSize(2)*0.25).*rand());
        else
            rndX = floor(imgSize(1)*0.65 + (imgSize(1)*0.7-imgSize(1)*0.3).*rand());
            rndY = floor(imgSize(2)*0.4 + (imgSize(2)*0.7-imgSize(2)*0.25).*rand());          
        end
       
        center = [rndX rndY rndZ];         
        
        
        %if the patch of the ground truth with the center randomly selected
        %contains a microbleed reject it
        patchGT = resampled_gt(rndX-auxPatchSize(1):rndX + auxPatchSize(1)-1 , rndY-auxPatchSize(2):rndY+auxPatchSize(2)-1,rndZ-auxPatchSize(3):rndZ + auxPatchSize(3)-1);
        value = sum(sum(sum(patchGT)));
        %If value >0 in the mask then there is a microbleed or part of it
        %over there. And also check if the patch is already saved, we dont
        %want to repeat. 
        if (value == 0 && (ismember(center,center_list,'rows')==0))
            
            patch = resampled_img(rndX-auxPatchSize(1):rndX + auxPatchSize(1)-1 , rndY-auxPatchSize(2):rndY+auxPatchSize(2)-1,rndZ-auxPatchSize(3):rndZ + auxPatchSize(3)-1);
            patchFlatten=single(reshape(patch,[1,prod(patch_size)]));
            patchFlatten = (patchFlatten - min(patchFlatten(:)))./(max(patchFlatten(:)) - min(patchFlatten(:)));
            save(char(strcat(save_non_microbleeds_path,char(string(counter2)))),'patchFlatten','-v7.3')
            counter = counter + 1;
            counter2=counter2+1;
            center_list=[center_list;center];
        else

        end
        
        
    end

end

