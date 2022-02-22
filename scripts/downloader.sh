#!/bin/bash

wget "https://www.dropbox.com/sh/5frv7bkl5ufqxx7/AABvbh_w5dPMHAbJpEJhgqHza?dl=1" -O storage/data/splits.zip;
mkdir storage/data/vl-t5-splits;
unzip storage/data/splits.zip -d storage/data/vl-t5-splits/;
rm storage/data/splits.zip;
