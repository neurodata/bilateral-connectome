SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE_DIR=$SCRIPT_DIR/..

export RESAVE_DATA=True

cat $BASE_DIR/scripts/manifest.txt | xargs -I % sh $SCRIPT_DIR/make_notebook.sh %

# $SCRIPT_DIR/copy_results.sh
