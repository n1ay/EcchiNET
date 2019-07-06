#!/bin/bash
if [ $# -ne 1 ]; then
    echo 'Needs one argument only: output filename'
    exit -1
fi

output_filname="$1"; shift
cmd="ffmpeg"
video_filenames=$(ls -1 | grep '.mp4$' | grep 'tmp.vid')
video_count=0
for video_filename in ${video_filenames}; do
    cmd="$cmd -i ${video_filename}"
    ((video_count++))
done

if [[ ${video_count} -eq 0 ]]; then
    echo 'No videos found. Exiting.'
    exit -2
fi

cmd="${cmd} -filter_complex \""

for (( i = 0; i < ${video_count}; ++i )); do
    if [[ ${i} -eq 0 ]]; then
        cmd="${cmd}[${i}:v] [${i}:a]"
    else
        cmd="${cmd} [${i}:v] [${i}:a]"
    fi
done

cmd="${cmd}concat=n=${video_count}:v=1:a=1 [v] [a]\" -map \"[v]\" -map \"[a]\" -y ${output_filname}"

echo ${cmd}
eval ${cmd}