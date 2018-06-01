%% Data Augmentation
addpath('/home/javier/Documents/DOCUMENTOS/Microbleeds/Paper/cmb-3dcnn-code-v1.0/code/NIfTI_20140122');

% img_data_path = '~/Documentos/Microbleeds/Paper/raw_data/';
% save_datasets_path = '~/Imágenes/ImagesPreprocessing/';
%img_data_path='/home/javier/Documents/DOCUMENTOS/Microbleeds/GoDARTS/GTandData/';
resampled_img_path='/home/javier/Documents/DOCUMENTOS/Microbleeds/GoDARTS/DataResampledV2/';
resampled_GT_path='/home/javier/Documents/DOCUMENTOS/Microbleeds/GoDARTS/gtResampledV2/';
original_GT_path='/home/javier/Documents/DOCUMENTOS/Microbleeds/GoDARTS/gt/';
save_path='/home/javier/Documents/DOCUMENTOS/Microbleeds/GoDARTS/centersResampled/';
save_patches_path='/home/javier/Documents/DOCUMENTOS/Microbleeds/GoDARTS/PatchesWithMicrobleeds/';
augmented_data_path =  '/home/javier/Documents/DOCUMENTOS/Microbleeds/GoDARTS/PatchedMicrobleedAugmented/';


patch_size=[16 16 10];


resampled_img_files= dir(resampled_img_path);
centers_files = dir(save_path);

resampled_img_files(1:2)=[];%To delete the first to elements that are junk
centers_files(1:2)=[];
centers_files(end)=[];
l1=length(resampled_img_files);
l2=length(centers_files);


if (isequal(l1,l2)==0)
    fprintf ( 2, 'Error! There is not the same number of files in the directories!\n' );
    fprintf ( 2, 'Closing the Script. Retry again!\n' );
    return
end

counter=1;
auxPatchSize=patch_size/2;
grades= [-90 -180 -270];
for kk = 1:l1
    counter=1;
    %Load the image to take the patches from 
    nii_resampled_img = load_untouch_nii([resampled_img_path resampled_img_files(kk).name]);
    currentImage=nii_resampled_img.img;
    save_patches_path_mod = strcat(augmented_data_path,char(string(kk)),'/');
    if ~exist(save_patches_path_mod)
        mkdir(save_patches_path_mod);
    end
    %load the centers for the current image
    
    load(strcat(save_path,char(string(kk)),'.mat'),'-mat');
    
    for ii=1:size(all_centers,1)
        currentCenter=all_centers(ii,:);
        patch=currentImage(currentCenter(1)-auxPatchSize(1):currentCenter(1)+auxPatchSize(1)-1,currentCenter(2)-auxPatchSize(2):currentCenter(2)+auxPatchSize(2)-1,currentCenter(3)-auxPatchSize(3):currentCenter(3)+auxPatchSize(3)-1);
        patchFlatten=single(reshape(patch,[1,prod(patch_size)]));
        patchFlatten = (patchFlatten - min(patchFlatten(:)))./(max(patchFlatten(:)) - min(patchFlatten(:)));
        patch=reshape(patchFlatten,[16 16 10]);
        
        for i=0 -2:2:2
            for j=0 -2:2:2
                if (i==0  && j==0)
                else
                    modified_patch=currentImage(currentCenter(1)+(i)-auxPatchSize(1):currentCenter(1)+(i)+auxPatchSize(1)-1,currentCenter(2)+(j)-auxPatchSize(2):currentCenter(2)+(j)+auxPatchSize(2)-1,currentCenter(3)-auxPatchSize(3):currentCenter(3)+auxPatchSize(3)-1);
                    
                    %Rotate the moved microbleeds
                    for i=1:size(grades,2)
                        modified_patch_rotated=imrotate(modified_patch,grades(i));
                        modified_patch_save=reshape(modified_patch_rotated,[1,prod(patch_size)]);                
                        save(char(strcat(save_patches_path_mod,char(string(counter)),'_',char(string(ii)))),'modified_patch_save','-v7.3')
                        counter=counter+1;
                    end
                    
                    modified_patch_save=reshape(modified_patch,[1,prod(patch_size)]);                
                    save(char(strcat(save_patches_path_mod,char(string(counter)),'_',char(string(ii)))),'modified_patch_save','-v7.3');
                    counter=counter+1;
                end
            end
        end
        %Rotate the moved microbleeds
        for i=1:size(grades,2)
          modified_patch_rotated=imrotate(patch,grades(i));
          modified_patch_save=reshape(modified_patch_rotated,[1,prod(patch_size)]);                
          save(char(strcat(save_patches_path_mod,char(string(counter)),'_',char(string(ii)))),'modified_patch_save','-v7.3')
          counter=counter+1;
        end
        modified_patch=permute(patch,[2,1,3]);
        modified_patch_save=reshape(modified_patch,[1,prod(patch_size)]);                
        save(char(strcat(save_patches_path_mod,char(string(counter)),'_',char(string(ii)))),'modified_patch_save','-v7.3')
        counter=counter+1;
    end

    
end