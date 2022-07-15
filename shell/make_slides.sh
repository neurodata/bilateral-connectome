SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE_DIR=$SCRIPT_DIR/..

SLIDE_DIR=$BASE_DIR/docs/slides/$1
FILE=$SLIDE_DIR/$1.md

if test -f "$FILE"; then
    # echo "$FILE exists."
    marp --pdf --allow-local-files $FILE 
    marp --html --allow-local-files $FILE
else
    echo "$FILE does not exist."
fi

