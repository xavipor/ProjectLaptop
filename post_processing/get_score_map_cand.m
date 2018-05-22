function get_score_map_cand(result_path,dimx,dimy,dimz,threshold_score_mask)
    %threshold_score_mask: the threshold to obtain the candidates from score map, ranging between [0 1]
  
    score_map_path = [result_path 'score_map/'];              
    cand_path = [result_path 'score_map_cands/']; % the path of candidates from score map    
    M_layer = 1;
    
    if ~isdir(cand_path)
        mkdir(cand_path);
    end
    files = dir(score_map_path);
    files(1:2) = [];
    for jj = 1:length(files)      
        fprintf('Loading No.%d testing subject (total %d).\n', jj,length(files));        
        load([score_map_path num2str(jj) '_score_mask.mat']);
        sz_sp = size(score_mask);
        score_map = reshape(score_mask(2,:,:,:),sz_sp(2:end));
        filtered_score_map = peak_score_map(score_map);      
        [mask center_score_map] = get_proposal_from_score_map_all_count(filtered_score_map,threshold_score_mask);
        center = [2*M_layer*(center_score_map(:,1)-1)+dimx/2,2*M_layer*(center_score_map(:,2)-1)+dimy/2,2*M_layer*(center_score_map(:,3)-1)+dimz/2];
        save([cand_path num2str(jj) '_cand.mat'],'center');
    end       
end










