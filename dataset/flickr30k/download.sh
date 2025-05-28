#!/bin/bash

# Tải file ảnh
curl -L -o flickr30k-images.zip "https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr30k-images.zip"

# Tải annotation CSV
curl -L -o flickr_annotations_30k.csv "https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr_annotations_30k.csv"

# Giải nén
unzip -q flickr30k-images.zip -d ./

# Xóa file zip
rm flickr30k-images.zip

echo "Downloaded Flickr30k dataset successfully."
