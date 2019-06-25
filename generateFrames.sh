
INPUT_VIDEO=$1
OUTPUT_DIR=$2

base=$(basename "${INPUT_VIDEO}")
ffmpeg -i "${INPUT_VIDEO}" -q:v 1 -start_number 0 "${OUTPUT_DIR}"/"${base}_%03d.jpg"
