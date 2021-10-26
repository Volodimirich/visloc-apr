#!/bin/bash

mkdir data
wget - './data' --content-disposition http://140.238.217.97/index.php/s/VfnQ6TanxzBnpcP/download
unzip ./CambridgeLandmarks.zip -d ./data/


wget -P './weights' https://vision.in.tum.de/webshare/u/zhouq/visloc-apr/models/googlenet_places.extract.pth
