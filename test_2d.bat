set ModelName=svt
set SNRs=SNR5
set Target=echo_mnist_100.mat
set scatter=image_100.mat
@REM for %%r in (0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0) do (
    for %%r in (0.3) do (
    echo echo completion 
    python single_echo_completetion.py --model_name %ModelName% --model_dir ./%ModelName% --save_dir ./%ModelName%/mat_save --sample_rate %%r --echo_dir ../dataset/Aircraft/%SNRs% --item_name %Target% --echo_name input --test_num 1
    echo rma imaging
    python image_rma.py --echo_dir ./%ModelName%/mat_save/%%r --save_dir ./%ModelName%/image_rma/%%r --item_name %Target%
    echo fista imaging
    python image_fista.py --echo_dir ./%ModelName%/mat_save/%%r --save_dir ./%ModelName%/image_fista/%%r --item_name %Target%
    echo pefista imaging
    python image_pefista.py --echo_dir ./%ModelName%/mat_save/%%r --save_dir ./%ModelName%/image_pefista/%%r --item_name %Target%
    echo loss calculation
    python loss_diff_imag.py --complete_model %ModelName% --rma_dir ./%ModelName%/image_rma/%%r/image_result.mat --fista_dir ./%ModelName%/image_fista/%%r/image_result.mat --pefista_dir ./%ModelName%/image_pefista/%%r/image_result.mat --save_dir ./%ModelName%/loss_diff_imag/%%r --gt_scatter ../dataset/Aircraft/scatter/%scatter%
    @REM echo rma imaging svt
    @REM python image_rma.py --echo_dir ./%ModelName%/mat_save/%%r --save_dir ./%ModelName%/image_rma/%%r --item_name echo_mnist_1_svt.mat --save_name image_result_svt.mat
    @REM echo fista imaging svt
    @REM python image_fista.py --echo_dir ./%ModelName%/mat_save/%%r --save_dir ./%ModelName%/image_fista/%%r --item_name echo_mnist_1_svt.mat --save_name image_result_svt.mat
    @REM echo pefista imaging svt
    @REM python image_pefista.py --echo_dir ./%ModelName%/mat_save/%%r --save_dir ./%ModelName%/image_pefista/%%r --item_name echo_mnist_1_svt.mat --save_name image_result_svt.mat
    @REM echo loss calculation
    @REM python loss_diff_imag.py --complete_model %ModelName% --rma_dir ./%ModelName%/image_rma/%%r/image_result_svt.mat --fista_dir ./%ModelName%/image_fista/%%r/image_result_svt.mat --pefista_dir ./%ModelName%/image_pefista/%%r/image_result_svt.mat --save_dir ./%ModelName%/loss_diff_imag/%%r/svt
)

