#!/usr/bin/python3

import sys,os
from edos_ipr_analysis import collect_mobility_edges

global_ipr_cut = 0.0012
static_data_path = '../../Data/IPR-vs-Eigenvalues/Relaxed/'
edos_data_path = '../../Data/IPR-vs-Eigenvalues/'
mobility_edge_output = '../../Data/Mobility_Edge_Renorm/'

thermostats = ['Q', 'C']
samples = ['2ac', '3ac', '4ac', '7ac', '8ac', '10ac']
temperatures=[100, 200, 300, 400, 500]
collect_mobility_edges(input_data_path = edos_data_path, output_path = mobility_edge_output,\
                       thermostats = thermostats, samples = samples, temperatures = temperatures, ipr_cut=global_ipr_cut)

thermostats = ['Q','C']
samples = ['9ac']
temperatures = [100,250,500,750,1000]
collect_mobility_edges(input_data_path = edos_data_path, output_path = mobility_edge_output,\
                       thermostats = thermostats, samples = samples, temperatures = temperatures, ipr_cut=global_ipr_cut)

