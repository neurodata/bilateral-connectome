SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
BASE_DIR=$SCRIPT_DIR/..

SLIDE_DIR=$BASE_DIR/docs/slides/$1
FILE=$SLIDE_DIR/$1.md

if test -f "$FILE"; then
    echo "$FILE exists."
    # REF: https://github.com/orgs/marp-team/discussions/225
    # had timeout issues on some large presentations
    PUPPETEER_TIMEOUT=60000 marp --pdf --allow-local-files $FILE 
    marp --html --allow-local-files $FILE
else
    echo "$FILE does not exist."
fi

