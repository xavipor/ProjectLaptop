figure(1)
title('Patch plots');
for i=1:10
patchR = reshape(patchFlatten,[16 16 10]);
subplot(2,5,i);
imshow(patchR(:,:,i),[])
    
end
% 
% figure(2)
% title('Patch plots rotated');
% patchR = reshape(patchFlatten,[16 16 10]);
% for i=1:10
% 
% subplot(2,5,i);
% imshow(patchR(:,:,i),[])
%     
% end



