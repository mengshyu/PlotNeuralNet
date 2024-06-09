#!/bin/bash

bash ../tikzmake.sh mobilenet
pdftoppm mobilenet.pdf output -png
