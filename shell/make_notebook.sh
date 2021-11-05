build=false
while getopts 'b' opt; do
    case $opt in 
        b) build=true ;;
    esac
done

shift $((OPTIND -1))

jupytext --to notebook --output bilateral-connectome/docs/$1.ipynb bilateral-connectome/scripts/$1.py
jupyter nbconvert --to notebook --inplace --execute --ExecutePreprocessor.timeout=-1 bilateral-connectome/docs/$1.ipynb 
python bilateral-connectome/docs/add_cell_tags.py bilateral-connectome/docs/$1.ipynb

if "$build"; then
    jupyter-book build bilateral-connectome/docs
fi