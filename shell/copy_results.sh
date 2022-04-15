SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE_DIR=$SCRIPT_DIR/..

(cd $BASE_DIR/overleaf && git pull)
rsync -r --max-size=49m $BASE_DIR/results/figs $BASE_DIR/overleaf

rsync $BASE_DIR/results/glued_variables.txt $BASE_DIR/overleaf
(cd $BASE_DIR/overleaf && git add . && git commit -m 'update figures' && git push)