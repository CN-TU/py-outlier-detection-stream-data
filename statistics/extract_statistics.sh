mkdir paper_plots_and_tables

rm -rf ../tests/all_synthetic
mkdir ../tests/all_synthetic
cat ../tests/*/SUMMARY_xstream_T500.txt > ../tests/all_synthetic/SUMMARY_xstream_T500.temp
cat ../tests/*/SUMMARY_swknn_T500.txt > ../tests/all_synthetic/SUMMARY_swknn_T500.temp
cat ../tests/*/SUMMARY_swlof_T500.txt > ../tests/all_synthetic/SUMMARY_swlof_T500.temp
cat ../tests/*/SUMMARY_sdo_T500.txt > ../tests/all_synthetic/SUMMARY_sdo_T500.temp
cat ../tests/*/SUMMARY_sdostream_T500.txt > ../tests/all_synthetic/SUMMARY_sdostream_T500.temp
cat ../tests/*/SUMMARY_loda_T500.txt > ../tests/all_synthetic/SUMMARY_loda_T500.temp
cat ../tests/*/SUMMARY_rshash_T500.txt > ../tests/all_synthetic/SUMMARY_rshash_T500.temp
cat ../tests/*/SUMMARY_rrct_T500.txt > ../tests/all_synthetic/SUMMARY_rrct_T500.temp

mv ../tests/all_synthetic/SUMMARY_xstream_T500.temp ../tests/all_synthetic/SUMMARY_xstream_T500.txt
mv ../tests/all_synthetic/SUMMARY_swknn_T500.temp ../tests/all_synthetic/SUMMARY_swknn_T500.txt
mv ../tests/all_synthetic/SUMMARY_swlof_T500.temp ../tests/all_synthetic/SUMMARY_swlof_T500.txt
mv ../tests/all_synthetic/SUMMARY_sdo_T500.temp ../tests/all_synthetic/SUMMARY_sdo_T500.txt
mv ../tests/all_synthetic/SUMMARY_sdostream_T500.temp ../tests/all_synthetic/SUMMARY_sdostream_T500.txt
mv ../tests/all_synthetic/SUMMARY_loda_T500.temp ../tests/all_synthetic/SUMMARY_loda_T500.txt
mv ../tests/all_synthetic/SUMMARY_rshash_T500.temp ../tests/all_synthetic/SUMMARY_rshash_T500.txt
mv ../tests/all_synthetic/SUMMARY_rrct_T500.temp ../tests/all_synthetic/SUMMARY_rrct_T500.txt

bash run_boxplots.sh
bash run_cddiagrams.sh
bash run_perf_tables.sh
mv *.png paper_plots_and_tables
mv *.tex paper_plots_and_tables

