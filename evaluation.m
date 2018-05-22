function evaluation()

prediction = load('/media/haocheng/DATA_1T/CODE/cmb-3dcnn-code-v1.0/result_dundee_10/detection.mat');
cmb = prediction.cmb;

TP = 0;
FN = 0;
FP = 0;
for i = 1:1:length(cmb)
% for i = 1
    p = cmb{i};

    name = num2str(i);
    G = load(['/media/haocheng/DATA_1T/CODE/cmb-3dcnn-code-v1.0/code/gt_mat/'  name '.mat']);
    
%     if i < 10
%         name = ['0' num2str(i)];
%     else
%         name = num2str(i);
%     end
%     G = load(['/media/haocheng/DATA_1T/IMAGES/cmb-3dcnn-data/ground_truth/'  name '.mat']);

    g = G.cen;
    index = true(size(g,1),1);
    for j = 1:1:size(p,1)
        pl = p(j,:);
        D = pdist2( pl, g, 'euclidean');
        [minD, ind] = min(D);
        if minD < 10
%             TP = TP + 1;           
%             if index(ind) == false
%                 error('CMB has already been detected, something wrong')
%             else
%                 index(ind) = false;
%             end
             if index(ind) == false
%                 error('CMB has already been detected, something wrong')
                continue;
            else
                index(ind) = false;
                TP = TP + 1;
            end
            
        else
            FP = FP + 1;
        end        
    end
    FN = FN + sum(index);    
end

fprintf('Sensitivity: %0.4f\n', TP/(TP+FN));
fprintf('Precision: %0.4f\n', TP/(TP+FP));
fprintf('averFP: %0.4f\n', FP/i);