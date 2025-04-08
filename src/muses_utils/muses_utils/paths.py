v112_megacities = '/tb/CrIS/results/CRIS/Release_1.12.0/'
# Global_Survey_Grid_0.8  - uses CAMEL, Global_Survey_Grid_0.8_Emis_UWIS - uses UWIS (legacy emissivity)
v116_global_survey = '/project/muses/archive/CRIS/Release_1.16.0'

"""NB: there were run with CO2 in the BAR step, so they're not comparable to normal PAN retrievals!!!"""
susans_airs_test_20220318 = '/project/sandbox12/ssund/output/0000001720/Archive-2022-03-18-3pantests/'

"""
Emily created these netCDF files for us.
These are "Wofsy-sanctioned" profiles, with GEOS-Chem tacked on at top and bottom. The GEOS-Chem runs were done especially for the ATom campaign periods.

-Vivienne

Panther (ECD variable) measured on all 4 campaigns, GT-CIMS (?) didn't fly on ATom 1
Two sets of CO (QCLS & NOAA Picarro)
Columns: 'Year','Month','Day','Altm_meters','P_hPa', 'Lat','Lon', 'PAN_ECD_ppb', 'PAN_GTCIMS_ppb', 'CO_NOAA_ppb', 'CO_QCL_ppb','UTC_Mean_1s_Time'
Gridded on a common altitude grid.
"""
atom_profiles = '/tb/sandbox14/vpayne/retrievals/PAN/Fischer/ATom/profiles'
