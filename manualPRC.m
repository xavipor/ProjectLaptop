function manualPRC()

T = [0.05:0.05:0.95];
T = cat(2,T,[0.96:0.0001:1]);

result_path_1 = '/media/haocheng/DATA_1T/CODE/cmb-3dcnn-code-v1.0/result1/';
gt_path_1 = '/media/haocheng/DATA_1T/IMAGES/cmb-3dcnn-data/ground_truth/';

[R1, P1] = drawmanualPRC(result_path_1, gt_path_1, T, 1);
plot(R1, P1)

result_path_3 = '/media/haocheng/DATA_1T/CODE/cmb-3dcnn-code-v1.0/result_dundee_10/';
gt_path_3 = '/media/haocheng/DATA_1T/CODE/cmb-3dcnn-code-v1.0/code/gt_mat/';

[R3, P3] = drawmanualPRC(result_path_3, gt_path_3, T, 0);
plot(R3, P3)


end


function [R, P] = drawmanualPRC(result_path, gt_path, T, isTMI)

R = zeros(length(T),1);
P = zeros(length(T),1);

for n = 1:1:length(T)
    
    t = T(n);
    files = dir([result_path 'final_prediction']);
    files(1:2)=[];
    cmb = {};
    for i = 1:length(files)
        load([result_path 'final_prediction/' num2str(i) '_prediction.mat']);
        load([result_path 'score_map_cands/' num2str(i) '_cand.mat']);
        pred = find(prediction>t);
        pos = center(pred,:);
        dummy = [];
        for k = 1:size(pos,1)
            for l = k+1:size(pos,1)
                distance = norm((pos(k,:)-pos(l,:)),2);
                if distance < 10
                    dummy = [dummy k];
                end
            end
        end
        pos(dummy,:)=[];
        cmb{i} = pos;
    end

    TP = 0;
    FN = 0;
    FP = 0;
    for k = 1:1:length(cmb)
    % for i = 1
        p = cmb{k};
        
        if isTMI
             if k < 10
                name = ['0' num2str(k)];
            else
                name = num2str(k);
             end
        else
            name = num2str(k);
        end
        
        G = load([gt_path name '.mat']);
        g = G.cen;
        index = true(size(g,1),1);
        for j = 1:1:size(p,1)
            pl = p(j,:);
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
    R(n) =  TP/(TP+FN);
    P(n) = TP/(TP+FP);
end
end
    