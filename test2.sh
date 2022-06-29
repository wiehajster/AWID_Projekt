#!/bin/bash

echo Execute test2_ok.jl
julia test2_ok.jl
echo Execute test2_sk.py
python test2_sk.py
echo Execute test2_plot.py
python test2_plot.py