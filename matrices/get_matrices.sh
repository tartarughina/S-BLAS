#!/bin/bash

# Get archives
wget https://suitesparse-collection-website.herokuapp.com/MM/Williams/webbase-1M.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/s3dkq4m2.tar.gz

# Extract archives
tar -xvf webbase-1M.tar.gz
tar -xvf s3dkq4m2.tar.gz

# Move matrices
mv webbase-1M/webbase-1M.mtx .
mv s3dkq4m2/s3dkq4m2.mtx .

# Clean up
rm -rf webbase-1M.tar.gz
rm -rf s3dkq4m2.tar.gz
rm -rf webbase-1M
rm -rf s3dkq4m2
