
function filtered_score_map = peak_score_map(score_map)
    filtered_score_map = zeros(size(score_map));
    [maxmap maxidx] = minmaxfilt(score_map,[11,10,10],'max','same'); 
    bool = (score_map == maxmap);    
    filtered_score_map = single(bool).*score_map;
end