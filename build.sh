#!/bin/bash
apt-get update
apt-get install -y poppler-utils libblas-dev liblapack-dev rustc cargo
pip install -r requirements.txt
