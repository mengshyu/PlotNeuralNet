#!/bin/bash

source ~/workspace/venv/bin/activate
bash ../tikzmake.sh mobilenet
pdftoppm mobilenet.pdf output -png
