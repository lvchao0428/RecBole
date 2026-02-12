echo "run beauty tfidf"
nohup sh two_phase_run_tfidf_stratified.sh > two_phase_run_tfidf_stratified.log 2>&1 
echo "run beauty tfidf end"

echo "run beauty tfidf + llm"
nohup sh two_phase_run_tfidf_llm_stratified.sh > two_phase_run_tfidf_llm_stratified.log 2>&1 
echo "run beauty tfidf + llm end"

echo "run beauty multi-view 7b"
nohup sh two_phase_run_multiview_v2_stratified.sh > two_phase_run_multiview_v2_stratified.log 2>&1 
echo "run beauty multi-view 7b end"

echo "run beauty multi-view 14b"
nohup sh two_phase_run_multiview_v2_stratified_14b.sh > two_phase_run_multiview_v2_stratified_14b.log 2>&1 
echo "run beauty multi-view 14b end"



echo "run toy tfidf "
nohup sh two_phase_run_tfidf_toys_stratified.sh > two_phase_run_tfidf_toys_stratified.log 2>&1 
echo "run toy tfidf end"

echo "run toy tfidf + llm"
nohup sh two_phase_run_tfidf_llm_toys_stratified.sh > two_phase_run_tfidf_llm_toys_stratified.log 2>&1 
echo "run toy tfidf + llm end"

echo "run toy multi-view 7b"
nohup sh two_phase_run_multiview_v2_toys_stratified_7b.sh > two_phase_run_multiview_v2_toys_stratified_7b.log 2>&
echo "run toy multi-view 7b end"

echo "run toy multi-view 14b"
nohup sh two_phase_run_multiview_v2_toys_stratified_14b.sh > two_phase_run_multiview_v2_toys_stratified_14b.log 2>&
echo "run toy multi-view 14b end"
