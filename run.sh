export PATH="/usr/local/MATLAB/R2013b/bin/:$PATH"
matlab -nosplash -nodisplay -r pre_process

THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python fcn_wrapper.py

matlab -nosplash -nodisplay -r post_process

python 3d_cnn.py

matlab -nosplash -nodisplay -r final
matlab -nosplash -nodisplay -r evaluation
  



