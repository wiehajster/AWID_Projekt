#!/bin/bash

echo Execute test1_ns.jl
julia test1_code/test1_ns.jl
echo Execute test1_ok.jl
julia test1_code/test1_ok.jl
echo Execute test1_sk.py
python test1_code/test1_sk.py
echo Execute test1_plot.py
python test1_code/est1_plot.py
