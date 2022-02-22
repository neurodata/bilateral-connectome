dir=./bilateral-connectome/shell
$dir/make_notebook.sh define_data
$dir/make_notebook.sh show_data
$dir/make_notebook.sh er_unmatched_test
$dir/make_notebook.sh sbm_unmatched_test 
$dir/make_notebook.sh adjusted_sbm_unmatched_test # has a longer sim portion in it
$dir/make_notebook.sh kc_minus
$dir/make_notebook.sh rdpg_unmatched_test