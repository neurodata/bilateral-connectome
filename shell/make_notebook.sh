SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE_DIR=$SCRIPT_DIR/..

build=false
while getopts 'b' opt; do
    case $opt in 
        b) build=true ;;
    esac
done

shift $((OPTIND -1))

jupytext --to notebook --output $BASE_DIR/docs/$1.ipynb $BASE_DIR/scripts/$1.py
jupyter nbconvert --to notebook --inplace --execute --ExecutePreprocessor.timeout=-1 $BASE_DIR/docs/$1.ipynb 
python $BASE_DIR/docs/add_cell_tags.py $BASE_DIR/docs/$1.ipynb

if "$build"; then
    jupyter-book build $BASE_DIR/docs
fi