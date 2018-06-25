%% Check the number of paths 

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
save_non_microbleeds_path = '/home/javier/Documents/DOCUMENTOS/Microbleeds/GoDARTS/PatchesNoMicrobleeds/';
patch_size=[16 16 10];


resampled_img_files= dir(resampled_img_path);
resampled_gt_files = dir(resampled_GT_path);
original_gt_files=dir(original_GT_path);

resampled_img_files(1:2)=[];%To delete the first to elements that are junk
resampled_gt_files(1:2)=[];%To delete the first to elements that are junk
original_gt_files(1:3)=[];%ONE MORE HERE BECAUSE THERE WAS SOME JUNK FROM R

l1=length(resampled_img_files);
l2=length(resampled_gt_files);
l3=length(original_gt_files);

if (isequal(l1,l2,l3)==0)
    fprintf ( 2, 'Error! There is not the same number of files in the directories!\n' );
    fprintf ( 2, 'Closing the Script. Retry again!\n' );
    return
end
filterChoice=0.15;
output=[];
output_names=[];
for jj = 1:l1

    %fprintf('Loading No.%d %s subject (total %d).\n', jj,mode,num);
    nii_resampled_img = load_untouch_nii([resampled_img_path resampled_img_files(jj).name]);
    nii_resampled_gt = load_untouch_nii([resampled_GT_path resampled_gt_files(jj).name]);
    nii_original_gt = load_untouch_nii([original_GT_path original_gt_files(jj).name]);
    
    resampled_img = nii_resampled_img.img;
    resampled_gt = nii_resampled_gt.img ;
    original_gt = nii_original_gt.img;
    
    copy=resampled_gt;
    filter = find(copy<filterChoice);
    patches_original= bwconncomp (original_gt,6);
    patches_resampled= bwconncomp (resampled_gt,6);
    copy(filter)=0;
    patches_resampled_filtered=bwconncomp (copy,6);
    
    name=original_gt_files(jj).name;
    name_resampled = resampled_gt_files(jj).name;
    aux=[patches_original.NumObjects;patches_resampled.NumObjects;patches_resampled_filtered.NumObjects;string(name);string(name_resampled)];
    
    
    output=[output aux];

end

%Based on the results obtained in here, the best option it is to use the
%filtered version since have always the same amount of microbleeds than the
%original one without resampling. What is suspicious is the big number of
%microbleeds compared with the ones that are provided by the paper...



%% Start to crop patches with microbleeds by finding the center and then cut the path of size 16x16x10

for jj = 1:l1

    %Use only resampled with filter since as we see before is the one with
    %better results
    nii_resampled_img = load_untouch_nii([resampled_img_path resampled_img_files(jj).name]);
    nii_resampled_gt = load_untouch_nii([resampled_GT_path resampled_gt_files(jj).name]);
    
    resampled_img = nii_resampled_img.img;
    resampled_gt = nii_resampled_gt.img ;
    copy=resampled_gt;
    
    filter = find(copy<filterChoice);
    copy(filter)=0;
    [patches_resampled_filtered,numberFound]=bwlabeln (copy,6);

    %Now it is time to work with the coordinates of the mask ( where the
    %microbleeds are drawn and crop the patches from the original image.
    %First step will be to find the center of the microbleeds.
    all_centers=[];

    for ii = 1:numberFound
        
        current_block = find(patches_resampled_filtered == ii);
        
        %for the depth it is quite easy since we can find the first element
        %of the current block and the last one and we can easily find the
        %center by getting the middle point. 
        [~,~,z1] = ind2sub(size(copy),current_block(1));
        [~,~,z2] = ind2sub(size(copy),current_block(end));
        z=z1+floor((z2-z1)/2);       
        %Now it is possible to go to the layer specified by z and find the
        %central point in the other two dimensions. This way, we can assume
        %that the first point >0 that we encounter on that layer will be
        %the point on the left, because of the way that find works.
        %Perhapst to find the first rown and the last row, I can use sum()
        %and sum by columns so we end up with a new matrix with n rows and
        %1 colum and we can find the first value != 0 and the last one. 
        
        %Sum (asda,1) Colapsa sobre 1xm
        %sum (asda,2) Colapsa sobre nx1
        copy2 = patches_resampled_filtered == ii;
        auxlayer=copy2(:,:,z);
        auxCols=sum(auxlayer,1);%Here we can get the first and the last column with values (sum over rows)
        auxRows=sum(auxlayer,2);%Here we can get the first and last row with values. (sum over columns)
        
        [a,~]=find(auxRows>0);
        [~,b]=find(auxCols>0);
        y=b(1) + floor((b(end)-b(1))/2);
        x=a(1) + floor((a(end)-a(1))/2);
        center = [x y z];
        all_centers=[all_centers;center];
        
        size_image= size(resampled_img);
        %Check if the microbleed is too much in the borders, perhaps an
        %human error as it happens in image 6.
        fileID = fopen(strcat(save_path,'log.txt'),'at');
        if ( (size_image(1)*0.8 < center(1) || (size_image(1)*0.2 > center(1) )) || (size_image(2)*0.8 < center(2) || size_image(2)*0.2 > center(2) ) || (size_image(3)*0.8 < center(3) || size_image(3)*0.2 > center(3) ) )
           fprintf (fileID,'************************************************************') ;
           fprintf (fileID,'------> Possible human error when labelling!  In the Microbleed %d of the image %s!\n',ii,resampled_gt_files(jj).name);
           fprintf (fileID,'************************************************************\n') ;
        end
        if (length(a)>16 || length(b)>16 || (z2-z1)>10)
            fprintf ( 2, 'In the microbleed %d of the image %s!\n',ii,resampled_gt_files(jj).name);
            fprintf ( 2, 'The size of the Microbleed is bigger than the path.Size of X,Y,Z =(%d %d %d)\n\n\n',length(a),length(b),(z2-z1) );
            fprintf (fileID,'In the microbleed %d of the image %s!\n',ii,resampled_gt_files(jj).name);
            fprintf (fileID,'The size of the Microbleed is bigger than the path.Size of X,Y,Z =(%d %d %d)\n\n\n',length(a),length(b),(z2-z1) );
        end
        fclose(fileID);
    end
    %Save the previously computed matrix of centers. 
    if (jj==6)
       all_centers = all_centers(1:7,:); 
    end
    save(strcat(save_path,char(string(jj)),'.mat'),'all_centers','-v7.3')%estoy guardando una mierda no una mat file.
    
    


end

%% Extract Patches with Microbleeds based on the centers and the centers obtained with the previous step


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
for kk = 1:l1
    counter=1;
    %Load the image to take the patches from 
    nii_resampled_img = load_untouch_nii([resampled_img_path resampled_img_files(kk).name]);
    currentImage=nii_resampled_img.img;
    save_patches_path_mod = strcat(save_patches_path,char(string(kk)),'/');
    if ~exist(save_patches_path_mod)
        mkdir(save_patches_path_mod);
    end
    %load the centers for the current image
    
    load(strcat(save_path,char(string(kk)),'.mat'),'-mat');
    for ii=1:size(all_centers,1)
        currentCenter=all_centers(ii,:);
        patch=currentImage(currentCenter(1)-auxPatchSize(1):currentCenter(1)+auxPatchSize(1)-1,currentCenter(2)-auxPatchSize(2):currentCenter(2)+auxPatchSize(2)-1,currentCenter(3)-auxPatchSize(3):currentCenter(3)+auxPatchSize(3)-1);
        patchFlatten=reshape(patch,[1,prod(patch_size)]);
        patchFlatten = (patchFlatten - min(patchFlatten(:)))./(max(patchFlatten(:)) - min(patchFlatten(:)));
        save(char(strcat(save_patches_path_mod,char(string(counter)))),'patchFlatten','-v7.3');
        counter=counter+1;
    end

    
end

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
save_non_microbleeds_path = '/home/javier/Documents/DOCUMENTOS/Microbleeds/GoDARTS/PatchesNoMicrobleeds/';

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
                        patchFlatten = modified_patch_save;
                        patchFlatten = (patchFlatten - min(patchFlatten(:)))./(max(patchFlatten(:)) - min(patchFlatten(:)));
                        save(char(strcat(save_patches_path_mod,char(string(counter)),'_',char(string(ii)))),'patchFlatten','-v7.3')
                        counter=counter+1;
                    end
                    
                    modified_patch_save=reshape(modified_patch,[1,prod(patch_size)]);
                    patchFlatten = modified_patch_save;
                    patchFlatten = (patchFlatten - min(patchFlatten(:)))./(max(patchFlatten(:)) - min(patchFlatten(:)));
                    save(char(strcat(save_patches_path_mod,char(string(counter)),'_',char(string(ii)))),'patchFlatten','-v7.3');
                    counter=counter+1;
                end
            end
        end
        %Rotate the moved microbleeds
        for i=1:size(grades,2)
          modified_patch_rotated=imrotate(patch,grades(i));
          modified_patch_save=reshape(modified_patch_rotated,[1,prod(patch_size)]); 
          patchFlatten = modified_patch_save;
          patchFlatten = (patchFlatten - min(patchFlatten(:)))./(max(patchFlatten(:)) - min(patchFlatten(:)));
          save(char(strcat(save_patches_path_mod,char(string(counter)),'_',char(string(ii)))),'patchFlatten','-v7.3')
          counter=counter+1;
        end
        modified_patch=permute(patch,[2,1,3]);
        modified_patch_save=reshape(modified_patch,[1,prod(patch_size)]); 
        patchFlatten = modified_patch_save;
        patchFlatten = (patchFlatten - min(patchFlatten(:)))./(max(patchFlatten(:)) - min(patchFlatten(:)));
        save(char(strcat(save_patches_path_mod,char(string(counter)),'_',char(string(ii)))),'patchFlatten','-v7.3')
        counter=counter+1;
    end

    
end
%% Generate patches without microbleeds.  This version divides the patches in folders, I am going to cjange this to just put all toguether without division of folder. 

resampled_img_files= dir(resampled_img_path);
resampled_gt_files = dir(resampled_GT_path);

resampled_img_files(1:2)=[];%To delete the first to elements that are junk
resampled_gt_files(1:2)=[];%To delete the first to elements that are junk

auxPatchSize=[16 16 10]/2;
center_list=[0,0,0];
l1=length(resampled_img_files);

for jj = 1:l1
    
    save_non_microbleeds_path_mod = strcat(save_non_microbleeds_path,char(string(jj)),'/');
    if ~exist(save_non_microbleeds_path_mod)
        mkdir(save_non_microbleeds_path_mod);
    end
    
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
            save(char(strcat(save_non_microbleeds_path_mod,char(string(counter)))),'patchFlatten','-v7.3')
            counter = counter + 1;
            center_list=[center_list;center];
        else

        end
        
        
    end

end
%% Save all the patches in the same folder. 
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
