
echo "1. two_phase_run_multiview_v3_toys_stratified_7b_nosenet_nocross.sh"
nohup sh two_phase_run_multiview_v3_toys_stratified_7b_nosenet_nocross.sh > two_phase_run_multiview_v3_toys_stratified_7b_nosenet_nocross.log 2>&1
echo "1. end"

echo "2. two_phase_run_multiview_v3_toys_stratified_7b_nosenet.sh"
nohup sh two_phase_run_multiview_v3_toys_stratified_7b_nosenet.sh > two_phase_run_multiview_v3_toys_stratified_7b_nosenet.log 2>&1
echo "2. end"

echo "3. two_phase_run_multiview_v3_toys_stratified_7b_nocross.sh"
nohup sh two_phase_run_multiview_v3_toys_stratified_7b_nocross.sh > two_phase_run_multiview_v3_toys_stratified_7b_nocross.log 2>&1
echo "3. end"

echo "4. two_phase_run_multiview_v3_toys_stratified_7b_no_whiten.sh"
nohup sh two_phase_run_multiview_v3_toys_stratified_7b_no_whiten.sh > two_phase_run_multiview_v3_toys_stratified_7b_no_whiten.log 2>&1
echo "4. end"

echo "5. two_phase_run_tfidf_llm_v3_toys_stratified.sh"
nohup sh two_phase_run_tfidf_llm_v3_toys_stratified.sh > two_phase_run_tfidf_llm_v3_toys_stratified.log 2>&
echo "5. end"
