function [mask center] = get_proposal_from_score_map_all_count(score_map,prob_threshold)    
    center = [];
    mask = logical(zeros(size(score_map)));    
    mask(score_map > prob_threshold) = 1;    
    [x y z] = ind2sub(size(mask),find(mask==1));
    center = [x y z];
end