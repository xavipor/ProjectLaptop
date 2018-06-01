resampled_img_path='/home/javier/Documents/DOCUMENTOS/Microbleeds/GoDARTS/DataResampled/';
resampled_GT_path='/home/javier/Documents/DOCUMENTOS/Microbleeds/GoDARTS/gtResampled/';
original_GT_path='/home/javier/Documents/DOCUMENTOS/Microbleeds/GoDARTS/gt/';
save_path='/home/javier/Documents/DOCUMENTOS/Microbleeds/GoDARTS/centersResampled/';
save_patches_path='/home/javier/Documents/DOCUMENTOS/Microbleeds/GoDARTS/PatchesWithMicrobleeds/';
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
for jj = 1:l1
    %Load the image to take the patches from 
    nii_resampled_img = load_untouch_nii([resampled_img_path resampled_img_files(jj).name]);
    currentImage=nii_resampled_img.img;
    
    %load the centers for the current image
    load(strcat(save_path,centers_files(jj).name),'-mat');
    for ii=1:size(all_centers,1)
        currentCenter=all_centers(ii,:);
        patch=currentImage(currentCenter(1)-auxPatchSize(1):currentCenter(1)+auxPatchSize(1)-1,currentCenter(2)-auxPatchSize(2):currentCenter(2)+auxPatchSize(2)-1,currentCenter(3)-auxPatchSize(3):currentCenter(3)+auxPatchSize(3)-1);
        save(char(strcat(save_patches_path,string(counter))),'patch');
        counter=counter+1;
    end

    
end
    