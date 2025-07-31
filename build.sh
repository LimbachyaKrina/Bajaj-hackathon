#!/bin/bash
apt-get update
apt-get install -y poppler-utils libblas-dev liblapack-dev rustc cargo
export CARGO_HOME=/tmp/cargo
export RUSTUP_HOME=/tmp/rustup
pip install cryptography --no-build-isolation
pip install -r requirements.txt
