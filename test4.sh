#!/bin/bash

echo Execute test4_ns.jl
julia test4_ns.jl
echo Execute test4_ok.jl
julia test4_ok.jl
echo Execute test4_sk.py
python test4_sk.py
#echo Execute test4_cnn.py
#python test4_cnn.py
echo Execute test4_plot.py
python test4_plot.py