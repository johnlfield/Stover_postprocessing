# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Post-processing DayCent corn stover regional simulation results
# This Jupyter Notebook is designed to facilitate post-processing and analysis of sets of raw DayCent results from a regional scale simulation. For more information, contact author [John Field](https://johnlfield.weebly.com/) at <john.L.field@gmail.com>
#
# ## DayCent background
# DayCent is a process-based model that simulates agro-ecosystem net primary production, soil organic matter dynamics, and nitrogen (N) cycling and trace gas emissions. DayCent is a daily-timestep version of the older CENTURY model. Both models were created and are currently maintained at the Colorado State University [Natural Resource Ecology Laboratory](https://www.nrel.colostate.edu/) (CSU-NREL), and source code is available upon request. 
# ![Alt text](DayCent.png)
#
# DayCent model homepage:  [https://www2.nrel.colostate.edu/projects/daycent/](https://www2.nrel.colostate.edu/projects/daycent/)
#
# In bioenergy sustainability studies, DayCent is typically used to estimate:
# * biomass yields
# * annual emissions of nitrous oxide (N2O), a potent greenhouse gas (GHG) generated in soils from synthetic and organic nitrogen fertilizer
# * changes in soil organic carbon (SOC) levels over time
#
# ## Regional simulation workflow
# The primary spatial data inputs to DayCent are:
# * soil texture as a function of depth
# * historic daily weather (Tmin, Tmax, precip)
#
# Our DayCent spatial modeling workflow is based on a national-scale GIS database of current land use ([NLCD](https://www.mrlc.gov/national-land-cover-database-nlcd-2016)), soil ([SSURGO](https://www.nrcs.usda.gov/wps/portal/nrcs/detail/soils/survey/?cid=nrcs142p2_053627)), and weather ([NARR](https://www.ncdc.noaa.gov/data-access/model-data/model-datasets/north-american-regional-reanalysis-narr)) data layers housed at CSU-NREL. The python-based workflow consists of a collection of scripts that perform the following:
# 1. Selection of area to be simulated, specified based on current land cover and/or land biophysical factors (i.e., soil texutre, slope, land capability class rating, etc.)
# 2. Determination of individual unique DayCent model runs (i.e., **"strata"**) necessary to cover the heterogenity of soils and climate across the simulation area
# 3. Parallel execution of simulations on the CSU-NREL computing cluster
# 4. Results analysis and mapping (this routine)
#

# import the necessary modules
import constants
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
import plotly as py
import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, iplot
from plotly.graph_objs import *

# ## Loading data
# The code below loads partially-analyzed DayCent simulation results at the county scale into Pandas dataframes for furhter analysis. Results are included for four different management scenarios that were simulated:
# * no stover removal (G, i.e., grain harvest only)
# * 25% stover harvest (G25S)
# * 50% stover harvest (G50S)
# * 75% stover harvest (G75S)
#
# Simulation results are spread across multiple .csv-format results files as follows:
# * corn–soy area per county (area_fips_data.csv)
# * annual simulated corn yields per county (corn_yield_year_county.csv)
# * annual simulated soy yields per county (soybean_yield_year_county.csv)
# * annual simulated stover yields per county (Stover_yield_year_county.csv)
# * annual simulated SOC levels per county (SOC_year_county.csv)
# * annual simulated N2O emissions per county (N2O_year_county.csv)

area_df = pd.read_csv("area_fips_data.csv", usecols=[1, 2])
corn_df = pd.read_csv("corn_yield_year_county.csv", usecols=[1, 2, 3, 4])
soy_df = pd.read_csv("soybean_yield_year_county.csv", usecols=[1, 2, 3, 4])
stover_df = pd.read_csv("Stover_yield_year_county.csv", usecols=[1, 2, 3, 4])
soc_df = pd.read_csv("SOC_year_county.csv", usecols=[1, 2, 3, 4])
n2o_df = pd.read_csv("N2O_year_county.csv", usecols=[1, 2, 3, 4])
soc_df

# ## Unit conversions
# In the following operations I'm updating to more formal units using the conversion factors in constants.py. I keep track of the updated units in the column names. For simplicity, after each conversion, I drop the original data in now-obsolete units.

# +
area_df["area_ha"] = area_df["area_acres"] * constants.ha_per_ACRE
area_df.drop(columns=["area_acres"], inplace=True)

corn_df["corn_yield_Mg_ha-1"] = corn_df["grainyield_bu_ac"] * ((constants.kg_per_bu_CORN * 0.001) / constants.ha_per_ACRE)
corn_df.drop(columns=["grainyield_bu_ac"], inplace=True)

soy_df["soy_yield_Mg_ha-1"] = soy_df["grainyield_bu_ac"] * ((constants.kg_per_bu_SOY * 0.001) / constants.ha_per_ACRE)
soy_df.drop(columns=["grainyield_bu_ac"], inplace=True)

stover_df["stover_yield_Mg_ha-1"] = stover_df["stover_dryyield_kgha"] * 0.001
stover_df.drop(columns=["stover_dryyield_kgha"], inplace=True)

soc_df["SOC_MgC_ha-1"] = soc_df["SOC_20cm_g_m2"] * constants.g_m2_to_Mg_ha
soc_df.drop(columns=["SOC_20cm_g_m2"], inplace=True)

n2o_df["N2O_MgN_ha-1"] = n2o_df["N2O_gN_m2"] * constants.g_m2_to_Mg_ha
n2o_df.drop(columns=["N2O_gN_m2"], inplace=True)

soc_df
# -

# ## Computing annual SOC differences
# I implemented this by ordering the data by fips|treatment|time, computing row differences, and then dropping the first year of each series (which reflects a difference between treatments instead of between years).

soc_df = soc_df.sort_values(["fips", "stover_removal", "simyear"])
soc_df["dSOC_MgC_ha-1"] = soc_df["SOC_MgC_ha-1"].diff()
soc_df = soc_df[(soc_df["simyear"] != 2011)]
soc_df

# ## Data aggregation & merges
# Here I calculate mean yields, N2O emissions & SOC changes over the full course of the simulation, and then merge all results together into a single data frame. Note that the aggregation has to come before the merges, since corn & soy are harvested in alternate years and thus cannot be merged on 'simyear'. Also, note that all year 2011 yield results are dropped on merge, since there are no dSOC results for that year.

# +
corn_df = corn_df[["fips", "stover_removal", "corn_yield_Mg_ha-1"]].groupby(["fips", "stover_removal"]).mean()
soy_df = soy_df[["fips", "stover_removal", "soy_yield_Mg_ha-1"]].groupby(["fips", "stover_removal"]).mean()
stover_df = stover_df[["fips", "stover_removal", "stover_yield_Mg_ha-1"]].groupby(["fips", "stover_removal"]).mean()
soc_df = soc_df[["fips", "stover_removal", "dSOC_MgC_ha-1"]].groupby(["fips", "stover_removal"]).mean()
n2o_df = n2o_df[["fips", "stover_removal", "N2O_MgN_ha-1"]].groupby(["fips", "stover_removal"]).mean()

df = pd.merge(corn_df, soy_df, on=['fips', 'stover_removal'])
df = pd.merge(df, stover_df, on=['fips', 'stover_removal'])
df = pd.merge(df, soc_df, on=['fips', 'stover_removal'])
df = pd.merge(df, n2o_df, on=['fips', 'stover_removal'])
df.reset_index(inplace=True)
df
# -

# ## Pivot
# This operation re-shapes the data such that the results for different management treatments (e.g., G, G25S, etc.) are shown in different columns instead of different rows. After the pivot step, we compute SOC and N2O differences between treatments. 

pivoted_df = df.pivot(index='fips', columns='stover_removal')
pivoted_df

pivoted_df.index
pivoted_df['dSOC_G25S_relative'] = pivoted_df['dSOC_MgC_ha-1']['G25S'] - pivoted_df['dSOC_MgC_ha-1']['G']
pivoted_df['dSOC_G50S_relative'] = pivoted_df['dSOC_MgC_ha-1']['G50S'] - pivoted_df['dSOC_MgC_ha-1']['G']
pivoted_df['dSOC_G75S_relative'] = pivoted_df['dSOC_MgC_ha-1']['G75S'] - pivoted_df['dSOC_MgC_ha-1']['G']
pivoted_df['dN2O_G25S_relative'] = pivoted_df['N2O_MgN_ha-1']['G25S'] - pivoted_df['N2O_MgN_ha-1']['G']
pivoted_df['dN2O_G50S_relative'] = pivoted_df['N2O_MgN_ha-1']['G50S'] - pivoted_df['N2O_MgN_ha-1']['G']
pivoted_df['dN2O_G75S_relative'] = pivoted_df['N2O_MgN_ha-1']['G75S'] - pivoted_df['N2O_MgN_ha-1']['G']
pivoted_df['dSOC_MgStover'] = pivoted_df['dSOC_G25S_relative'] / pivoted_df['stover_yield_Mg_ha-1']['G25S']
pivoted_df

# ## Remaining operations
# * compute net CO2e biogenic GHG footprint
# * pivot and calculate relative results for each scenario
# * save select results to file

# ## Mapping our results
# Below we use the Plotly module "choropleth" tool to map select county-scale results. The fips_mapping() function helps to standardize the formatting, color scheme, and scale for each map.  

# +
init_notebook_mode(connected=True)

# scope = ''
scope = ['ND', 'SD', 'NE', 'KS', 'MO', 'IA', 'MN', 'WI', 'IL', 'KY', 'IN', 'MI', 'OH', 'PA', 'WV', 'MD', 'DE',
         'NY', 'TN', 'AR', 'OK', 'VA', 'NC']

def fips_mapping(fips, data, title, legend_title, linspacing, divergent=False, reverse=False):
   
    # use 'linspacing' parameters to create a bin list, and specify rounding if values are small-ish
    bin_list = np.linspace(linspacing[0], linspacing[1], linspacing[2]).tolist()
    rounding = True
    if linspacing[1] < 10:
        rounding = False

    kwargs = {}
    if scope:
        kwargs['scope'] = scope

    if divergent:
        # convert matplotlib (r, g, b, x) tuple color format to 'rgb(r, g, b)' Plotly string format
        cmap = plt.get_cmap('RdBu')  # or RdYlBu for better differentiation vs. missing data squares in tiling map
        custom_rgb_cmap = [cmap(x) for x in np.linspace(0, 1, (linspacing[2] + 1))]
        custom_plotly_cmap = []
        for code in custom_rgb_cmap:
            plotly_code = 'rgb({},{},{})'.format(code[0] * 255.0, code[1] * 255.0, code[2] * 255.0)
            custom_plotly_cmap.append(plotly_code)
        if reverse:
            custom_plotly_cmap.reverse()

        kwargs['state_outline'] = {'color': 'rgb(100,100,100)', 'width': 1.0}
        kwargs['colorscale'] = custom_plotly_cmap

    fig = ff.create_choropleth(fips=fips.tolist(),
                               values=data.tolist(),
                               binning_endpoints=bin_list,
                               round_legend_values=rounding,
                               county_outline={'color': 'rgb(255,255,255)', 'width': 0.25},
                               legend_title=legend_title,
                               title=title,
                               paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(0,0,0,0)',
                               **kwargs)
    iplot(fig)


# -

fips_mapping(pivoted_df.index, pivoted_df['stover_yield_Mg_ha-1']['G25S'], "Stover yield @ 25% removal rate", '(Mg ha-1)', (2, 4, 21))

fips_mapping(pivoted_df.index, pivoted_df['dSOC_MgStover'], "SOC penalty per mass of stover harvested", '(Mg C (Mg biomass)-1)', (-.05, 0.05, 41), divergent=True)


