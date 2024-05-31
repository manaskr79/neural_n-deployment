#!/bin/bash

sudo apt install python3-pip

sudo apt install python3-venv

python -m venv .venv

source .venv/bin/bash/activate

pip3 install -r requirements.txt

code .
