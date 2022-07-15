#!/bin/bash

# Assume parameter passed in is a relative path to a directory.
# For brevity, we won't do argument type or length checking.

ABS_PATH=`cd "$1"; pwd` # double quotes for paths that contain spaces etc...
echo "Absolute path: $ABS_PATH"