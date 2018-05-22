function drawPRCs____()

% result_path_1 = '/media/haocheng/DATA_1T/CODE/cmb-3dcnn-code-v1.0/result1/';
% gt_path_1 = '/media/haocheng/DATA_1T/IMAGES/cmb-3dcnn-data/ground_truth/';
% 
% Draw_Precision_Recall_Curve(result_path_1, gt_path_1, 1);

% 
% result_path_2 = '/media/haocheng/DATA_1T/CODE/cmb-3dcnn-code-v1.0/result_raw_data_upsampled/';
% gt_path_2 = '/media/haocheng/DATA_1T/IMAGES/cmb-3dcnn-data/ground_truth/';
% 
% Draw_Precision_Recall_Curve(result_path_2, gt_path_2, 1);

result_path_3 = '/media/haocheng/DATA_1T/CODE/cmb-3dcnn-code-v1.0/result_dundee_10/';
gt_path_3 = '/media/haocheng/DATA_1T/CODE/cmb-3dcnn-code-v1.0/code/gt_mat/';
% 
Draw_Precision_Recall_Curve(result_path_3, gt_path_3, 0);

% plot(R1,P1,'b', R2, P2, 'g', R3, P3, 'r')
% 
% legend(['TMI-20   AUC:' num2str(info1.auc,2)],['TMI-20-down/upsampled   AUC:' num2str(info2.auc,2)], ['Tayside-10   AUC:' num2str(info3.auc,2)], 'Location', 'southwest');
% xlabel('Recall(Sensitivity)');
% ylabel('Precision');




end



function Draw_Precision_Recall_Curve(result_path, gt_path, isTMI)


files = dir([result_path 'final_prediction']);
files(1:2)=[];


TP = 0;
FN = 0;
FP = 0;

for i = 1:1:length(files)
% for i = 3
    
    if isTMI
        if i < 10
            name = ['0' num2str(i)];
        else
            name = num2str(i);
        end
    else
        name = num2str(i);
    end
    
    % load probabilities
    load([result_path 'final_prediction/' num2str(i) '_prediction.mat']);
    
    % load location
    load([result_path 'score_map_cands/' num2str(i) '_cand.mat']);
    
    % load ground truth
    G = load([gt_path  name '.mat']);
    g = G.cen;
    

    pre = find(prediction>0.85);
    pos = center(pre,:);
    
%   exclude same points
    dummy = [];
    for k = 1:size(pos,1)
        for l = k+1:size(pos,1)
            distance = norm((pos(k,:)-pos(l,:)),2);
            if distance < 10
                dummy = [dummy k];
            end
        end
    end
    
    pos(dummy,:) = [];
    
%     TPx = 0;
%     index = true(size(g,1),1);
%     for j = 1:1:size(g,1)
%         gl = g(j,:);
%         D = pdist2(pos, gl, 'euclidean');
%         [minD, ind] = min(D);
%         if minD < 10
%             index(j) = false;
%             TPx = TPx + 1;           
%         end        
%     end
%     TP = TP + TPx;
%     FN = FN + sum(index);
%     FP = FP + length(pos) - TPx;
        
    index = true(size(g,1),1);
    for j = 1:1:size(pos,1)
        pl = pos(j,:);
        D = pdist2( pl, g, 'euclidean');
        [minD, ind] = min(D);
        if minD < 10
             if index(ind) == false
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

end

