#!/bin/bash

filename="$1"; shift
base_filename=$(echo "$filename" | awk -F'.mp4' '{ print $1 }')
preprocessed_filename="${base_filename}_preproc.mp4"

ffmpeg -i "$filename" -vf scale=240:180,format=gray -r 4 "$preprocessed_filename"
echo "Preprocessed file: $filename. Output saved to $preprocessed_filename."
