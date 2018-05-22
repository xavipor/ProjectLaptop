clear;clc;
addpath('/media/haocheng/DATA_1T/CODE/cmb-3dcnn-code-v1.0/code/post_processing');
addpath('/media/haocheng/DATA_1T/CODE/cmb-3dcnn-code-v1.0/code/NIfTI_20140122/')
addpath('/media/haocheng/DATA_1T/CODE/cmb-3dcnn-code-v1.0/code/lib/MinMaxFilterFolder');
result_path = '/media/haocheng/DATA_1T/CODE/cmb-3dcnn-code-v1.0/result1/';

%% get the score map candidates
dimx = 16;
dimy = 16;
dimz = 10;
threshold_score_mask = 0.64;
get_score_map_cand(result_path,dimx,dimy,dimz,threshold_score_mask);


%% get stage-2 patches
dimx = 20;
dimy = 20;
dimz = 16;
get_datasets_final_pred(result_path,dimx,dimy,dimz);
% get_datasets_final_pred_dundee(result_path,dimx,dimy,dimz)