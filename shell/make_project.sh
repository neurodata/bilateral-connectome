SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE_DIR=$SCRIPT_DIR/..


while getopts 'r' opt; do
    case $opt in 
        r) 
            export RERUN_SIMS=true
            export RESAVE_DATA=true
            echo "[make_project.sh] Set to rerun simulations and resaving data"
        ;;
    esac
done

shift $((OPTIND -1))

echo "[make_project.sh] Running notebooks via make_notebook.sh"
cat $BASE_DIR/scripts/manifest.txt | xargs -I % sh $SCRIPT_DIR/make_notebook.sh %

$SCRIPT_DIR/copy_results.sh
