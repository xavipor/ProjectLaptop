import os
from lib.wrap_test import test_wrapper
from lib.relu import relu
import theano
import sys

def test_model():
    
    para_path = '/home/xavipor/Documentos/Microbleeds/Paper/cmb-3dcnn-code-v1.0/demo/model/fine_tuned_params.pkl'
    result_path = '/home/xavipor/Documentos/Microbleeds/Paper/cmb-3dcnn-code-v1.0/demo/result/'
    
    patch_size = [16,16,10]
    input_sizes = [512,512,148] #[in_height,in_width,in_time]
    output_sizes = [249,249,70]       #Por que este output size?? de donde sale este valor?? ## calculado a basse de ver los filtros y los pooling
    # clip_rate = [3,3,2]  #the clip size must be exactly divided by the corresponding dimensions of the output_sizes     
    clip_rate = [3,3,2]
    layer_num = 5
    M_layer = 1
    maxpool_sizes = [(2,2,2),(1,1,1),(1,1,1),(1,1,1),(1,1,1)]
    activations = [relu,relu,relu,relu,None]
    dropout_rates = [0.2,0.3,0.3,0.3,0.3]

    print ('wrapping test dataset ... ')
    whole_volume_path = '../result/mat_data/'
    save_score_map_path = result_path + 'score_map/'
    if not os.path.exists(save_score_map_path):
        os.makedirs(save_score_map_path)
        print ('New folder created:',save_score_map_path)
               
    test_wrapper(
        input_sizes = input_sizes,
        output_sizes = output_sizes,
        patch_size = patch_size,
        clip_rate = clip_rate,
        M_layer = M_layer,
        layer_num = layer_num,
        maxpool_sizes = maxpool_sizes,
        activations = activations,
        dropout_rates = dropout_rates,
        para_path = para_path,
        save_score_map_path = save_score_map_path,
        whole_volume_path = whole_volume_path,
        mode = 'test')
            
if __name__ == '__main__':
    try:
        test_model()
    except KeyboardInterrupt:
        sys.exit()
