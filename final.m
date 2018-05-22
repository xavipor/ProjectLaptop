function final()

result_path = '/media/haocheng/DATA_1T/CODE/cmb-3dcnn-code-v1.0/result_dundee_10/';
files = dir([result_path 'final_prediction']);
files(1:2)=[];
cmb = {};
T = 0.99999;
for i = 2:length(files)
    load([result_path 'final_prediction/' num2str(i) '_prediction.mat']);
    load([result_path 'score_map_cands/' num2str(i) '_cand.mat']);
    pred = find(prediction>T);
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
save('/media/haocheng/DATA_1T/CODE/cmb-3dcnn-code-v1.0/result_dundee_10/detection.mat','cmb');

end
