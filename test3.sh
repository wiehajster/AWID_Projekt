#!/bin/bash

echo Execute test3_ns.jl
julia test3_code/test3_ns.jl
echo Execute test3_ok.jl
julia test3_code/test3_ok.jl
echo Execute test3_sk.py
python test3_code/test3_sk.py
echo Execute test3_plot.py
python test3_code/test3_plot.py
