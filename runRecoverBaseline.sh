echo "rum mv v3 baseline"
nohup sh run_multiview_v3_baseline_stratified.sh > run_multiview_v3_baseline_stratified.log 2>&1


echo "run sasercalignv3 baseline"
nohup sh run50epBase_v3_stratified.sh > run50epBase_v3_stratified.log 2>&1
