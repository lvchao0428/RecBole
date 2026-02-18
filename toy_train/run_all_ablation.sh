
echo "1. two_phase_run_multiview_v3_toys_stratified_7b_nosenet_nocross.sh"
#nohup sh two_phase_run_multiview_v3_toys_stratified_7b_nosenet_nocross.sh > two_phase_run_multiview_v3_toys_stratified_7b_nosenet_nocross.log 2>&1
echo "1. end"

echo "2. two_phase_run_multiview_v3_toys_stratified_7b_nosenet.sh"
#nohup sh two_phase_run_multiview_v3_toys_stratified_7b_nosenet.sh > two_phase_run_multiview_v3_toys_stratified_7b_nosenet.log 2>&1
echo "2. end"

echo "3. two_phase_run_multiview_v3_toys_stratified_7b_nocross.sh"
#nohup sh two_phase_run_multiview_v3_toys_stratified_7b_nocross.sh > two_phase_run_multiview_v3_toys_stratified_7b_nocross.log 2>&1
echo "3. end"

echo "4. two_phase_run_multiview_v3_toys_stratified_7b_no_whiten.sh"
nohup sh two_phase_run_multiview_v3_toys_stratified_7b_no_whiten.sh > two_phase_run_multiview_v3_toys_stratified_7b_no_whiten.log 2>&1
echo "4. end"

echo "5. two_phase_run_tfidf_llm_v3_toys_stratified_no_whiten.sh"
nohup sh two_phase_run_tfidf_llm_v3_toys_stratified_no_whiten.sh > two_phase_run_tfidf_llm_v3_toys_stratified_no_whiten.log 2>&1
echo "5. end"


cd ..

echo "exp_sensitivity_v3_align_005 begin"
nohup sh experiments/exp_sensitivity_v3_align_005.sh > exp_sensitivity_v3_align_005.log 2>&1
echo "exp_sensitivity_v3_align_005 end"

echo "exp_sensitivity_v3_align_015 begin"
nohup sh experiments/exp_sensitivity_v3_align_015.sh > exp_sensitivity_v3_align_015.log 2>&1
echo "exp_sensitivity_v3_align_015 end"


echo "exp_sensitivity_v3_align_020 begin"
nohup sh experiments/exp_sensitivity_v3_align_020.sh > exp_sensitivity_v3_align_020.log 2>&1
echo "exp_sensitivity_v3_align_020 end"


echo "#################### cold  ####################"

echo "exp_sensitivity_v3_cold_20 begin"
nohup sh experiments/exp_sensitivity_v3_cold_20.sh > exp_sensitivity_v3_cold_20.log 2>&1
echo "exp_sensitivity_v3_cold_20 end"

echo "exp_sensitivity_v3_cold_40 begin"
nohup sh experiments/exp_sensitivity_v3_cold_40.sh > exp_sensitivity_v3_cold_40.log 2>&1
echo "exp_sensitivity_v3_cold_40 end"


echo "exp_sensitivity_v3_cold_50 begin"
nohup sh experiments/exp_sensitivity_v3_cold_50.sh > exp_sensitivity_v3_cold_50.log 2>&1
echo "exp_sensitivity_v3_cold_50 end"


echo "####################### infer  #################"
echo "exp_sensitivity_v3_infer_03 begin"
nohup sh experiments/exp_sensitivity_v3_infer_03.sh > exp_sensitivity_v3_infer_03.log 2>&1
echo "exp_sensitivity_v3_infer_03 end"

echo "exp_sensitivity_v3_infer_09 begin"
nohup sh experiments/exp_sensitivity_v3_infer_09.sh > exp_sensitivity_v3_infer_09.log 2>&1
echo "exp_sensitivity_v3_infer_09 end"


echo "exp_sensitivity_v3_infer_12 begin"
nohup sh experiments/exp_sensitivity_v3_infer_12.sh > exp_sensitivity_v3_infer_12.log 2>&1
echo "exp_sensitivity_v3_infer_12 end"


echo "######################## tau ################"
echo "exp_sensitivity_v3_tau_003 begin"
nohup sh experiments/exp_sensitivity_v3_tau_003.sh > exp_sensitivity_v3_tau_003.log 2>&1
echo "exp_sensitivity_v3_tau_003 end"

echo "exp_sensitivity_v3_tau_007 begin"
nohup sh experiments/exp_sensitivity_v3_tau_007.sh > exp_sensitivity_v3_tau_007.log 2>&1
echo "exp_sensitivity_v3_tau_007 end"


echo "exp_sensitivity_v3_tau_010 begin"
nohup sh experiments/exp_sensitivity_v3_tau_010.sh > exp_sensitivity_v3_tau_010.log 2>&1
echo "exp_sensitivity_v3_tau_010 end"


echo "########################################"
