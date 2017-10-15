#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

mkdir -p $DIR/data
rm -r $DIR/data/*
set -e

wget -O $DIR/data/testdata.zip https://www.dropbox.com/s/qqvokdtalf0kgs2/2017_English_final.zip#
unzip $DIR/data/testdata.zip -d $DIR/data

F=$DIR/data/2017_English_final/DOWNLOAD

cp -r $F/* $DIR/data/

rm -r $DIR/data/2017_English_final
rm $DIR/data/testdata.zip
rm -r $DIR/data/__MACOSX
