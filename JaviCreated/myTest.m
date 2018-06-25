%% Check the number of paths 

addpath('/home/javier/Documents/DOCUMENTOS/Microbleeds/Paper/cmb-3dcnn-code-v1.0/code/NIfTI_20140122');

% img_data_path = '~/Documentos/Microbleeds/Paper/raw_data/';
% save_datasets_path = '~/Imágenes/ImagesPreprocessing/';
%img_data_path='/home/javier/Documents/DOCUMENTOS/Microbleeds/GoDARTS/GTandData/';
resampled_img_path='/home/javier/Documents/DOCUMENTOS/Microbleeds/GoDARTS/DataResampled/';
resampled_GT_path='/home/javier/Documents/DOCUMENTOS/Microbleeds/GoDARTS/gtResampled/';
original_GT_path='/home/javier/Documents/DOCUMENTOS/Microbleeds/GoDARTS/gt/';
save_path='/home/javier/Documents/DOCUMENTOS/Microbleeds/GoDARTS/centersResampled/';
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
filterChoice=0.3;
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
    patches_resampled_filtered=bwconncomp (copy,6);
    
    %Now it is time to work with the coordinates of the mask ( where the
    %microbleeds are drawn and crop the patches from the original image.
    %First step will be to find the center of the microbleeds.
    all_centers=[]
    for ii = 1:patches_resampled_filtered.NumObjects
        
        current_block = cell2mat(patches_resampled_filtered.PixelIdxList(ii));
        
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
        auxlayer=copy(:,:,z);
        auxCols=sum(auxlayer,1);%Here we can get the first and the last column with values (sum over rows)
        auxRows=sum(auxlayer,2);%Here we can get the first and last row with values. (sum over columns)
        
        [a,~]=find(auxRows>0);
        [~,b]=find(auxCols>0);
        y=b(1) + floor((b(end)-b(1))/2);
        x=a(1) + floor((a(end)-a(1))/2);
        center = [x y z];
        all_centers=[all_centers;center];
        if (length(a)>16 || length(b)>16 || (z2-z1)>10)
            fprintf ( 2, 'In the microbleed %d of the image %s!\n',ii,resampled_gt_files(jj).name);
            fprintf ( 2, 'The size of the Microbleed is bigger than the path.Size of X,Y,Z =(%d %d %d)\n\n\n',length(a),length(b),(z2-z1) );
            fileID = fopen(strcat(save_path,'log.txt'),'at');
            fprintf (fileID,'In the microbleed %d of the image %s!\n',ii,resampled_gt_files(jj).name);
            fprintf (fileID,'The size of the Microbleed is bigger than the path.Size of X,Y,Z =(%d %d %d)\n\n\n',length(a),length(b),(z2-z1) );
            fclose(fileID);
        end
 
    end
    %Save the previously computed matrix of centers. 
    save(strcat(save_path,resampled_gt_files(jj).name),'all_centers')
    
    


end

%% Extract Patches with Microbleeds based on the centers and the centers obtained with the previous step

resampled_img_files= dir(resampled_img_path);
centers_files = dir(save_path);

resampled_img_files(1:2)=[];%To delete the first to elements that are junk
original_gt_files(1:3)=[];%ONE MORE HERE BECAUSE THERE WAS SOME JUNK FROM R

l1=length(resampled_img_files);
l2=length(resampled_gt_files);
l3=length(original_gt_files);

if (isequal(l1,l2,l3)==0)
    fprintf ( 2, 'Error! There is not the same number of files in the directories!\n' );
    fprintf ( 2, 'Closing the Script. Retry again!\n' );
    return
end


