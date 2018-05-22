function drawPRs()

% result_path_1 = '/media/haocheng/DATA_1T/CODE/cmb-3dcnn-code-v1.0/result1/';
% gt_path_1 = '/media/haocheng/DATA_1T/IMAGES/cmb-3dcnn-data/ground_truth/';
% 
% [R1, P1, info1] = Draw_Precision_Recall_Curve(result_path_1, gt_path_1, 1);
% plot(R1, P1)
% hold on, plot(0.892,0.725, '*r')

% 
% result_path_2 = '/media/haocheng/DATA_1T/CODE/cmb-3dcnn-code-v1.0/result_raw_data_upsampled/';
% gt_path_2 = '/media/haocheng/DATA_1T/IMAGES/cmb-3dcnn-data/ground_truth/';
% 
% [R2, P2, info2] = Draw_Precision_Recall_Curve(result_path_2, gt_path_2, 1);

result_path_3 = '/media/haocheng/DATA_1T/CODE/cmb-3dcnn-code-v1.0/result_dundee_10/';
gt_path_3 = '/media/haocheng/DATA_1T/CODE/cmb-3dcnn-code-v1.0/code/gt_mat/';

[R3, P3, info3] = Draw_Precision_Recall_Curve(result_path_3, gt_path_3, 0);
plot(R3, P3)
hold on, plot(0.763, 0.269, '*r');

% plot(R1,P1,'b', R2, P2, 'g', R3, P3, 'r')
% 
% legend(['TMI-20   AUC:' num2str(info1.auc,2)],['TMI-20-down/upsampled   AUC:' num2str(info2.auc,2)], ['Tayside-10   AUC:' num2str(info3.auc,2)], 'Location', 'southwest');
% xlabel('Recall(Sensitivity)');
% ylabel('Precision');


end



function [RECALL, PRECISION, INFO] = Draw_Precision_Recall_Curve(result_path, gt_path, isTMI)


files = dir([result_path 'final_prediction']);
files(1:2)=[];


LABELS = [];
SCORES = [];

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
    
    pos  = center;
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
    prediction(dummy) = [];
    
    % get label for each candiate
    labels_ = zeros(1, size(pos,1));
    index = true(size(g,1),1);
    for j = 1:1:size(pos,1)
        pl = pos(j,:);
        D = pdist2(pl, g, 'euclidean');
        [minD, ind] = min(D);
        if minD < 10;
            if index(ind) == false
                continue;
            else
                labels_(j) = 1;
                index(ind) = false;
            end
        else
            labels_(j) = -1;
        end
    end  
    
    labels_ = cat(2, labels_, ones(1,sum(index)));
    scores_ = cat(2, prediction, zeros(1, sum(index)));
    
    LABELS = cat(1, LABELS, labels_');
    SCORES = cat(1, SCORES, scores_');

end

% 
pred = double(SCORES>0.85);
pred(pred==0) = -1;
% 
% 
[sensitivity, precision] = measure(pred, LABELS)


[RECALL, PRECISION, INFO] = vl_pr(LABELS, SCORES);


end

function [sensitivity, precision] = measure(pred,labels)

if size(pred) ~= size(labels)
    error('mismatch size')
end

confusionMat = zeros(2,2);

for i = 1:1:length(labels)
    gt = labels(i);
    pp = pred(i);
    
    if gt == -1 && pp == -1
        confusionMat(2,2) = confusionMat(2,2) + 1;
    end
    
    if gt == -1 && pp == 1
        confusionMat(2,1) = confusionMat(2,1) + 1;
    end
    
    if gt == 1 && pp == -1
        confusionMat(1,2) = confusionMat(1,2) + 1;
    end
    
    if gt == 1 && pp == 1
        confusionMat(1,1) = confusionMat(1,1) + 1;
    end
end
    

sensitivity = confusionMat(1,1)./ (confusionMat(1,1) + confusionMat(1,2));
precision = confusionMat(1,1) ./ (confusionMat(1,1) + confusionMat(2,1));

end