set ModelName=pu
for %%r in (0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0) do (
@REM for %%r in (0.3) do (
    echo echo completion 
    python single_echo_completetion.py --model_name %ModelName% --model_dir ./%ModelName% --save_dir ./%ModelName%/mat_save --sample_rate %%r
    echo rma imaging
    python image_rma.py --echo_dir ./%ModelName%/mat_save/%%r --save_dir ./%ModelName%/image_rma/%%r --item_name echo_mnist_1_svt.mat --save_name image_result_svt.mat
    echo fista imaging
    python image_fista.py --echo_dir ./%ModelName%/mat_save/%%r --save_dir ./%ModelName%/image_fista/%%r --item_name echo_mnist_1_svt.mat --save_name image_result_svt.mat
    echo pefista imaging
    python image_pefista.py --echo_dir ./%ModelName%/mat_save/%%r --save_dir ./%ModelName%/image_pefista/%%r --item_name echo_mnist_1_svt.mat --save_name image_result_svt.mat
    echo loss calculation
    python loss_diff_imag.py --complete_model %ModelName% --rma_dir ./%ModelName%/image_rma/%%r/image_result_svt.mat --fista_dir ./%ModelName%/image_fista/%%r/image_result_svt.mat --pefista_dir ./%ModelName%/image_pefista/%%r/image_result_svt.mat --save_dir ./%ModelName%/loss_diff_imag/%%r/svt
)
