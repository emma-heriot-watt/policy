#!/bin/bash

wget "https://www.dropbox.com/sh/m26a281y97ok6wm/AAALB7np-NwyFGsjr8aH-YUDa?dl=1" -O storage/data/splits.zip;
mkdir storage/data/vl-t5-splits;
unzip storage/data/splits.zip -d storage/data/vl-t5-splits/;
rm storage/data/splits.zip;
