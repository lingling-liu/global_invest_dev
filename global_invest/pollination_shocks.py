import os
import hazelbean as hb
import numpy as np
import pandas as pd
import logging
import dask
import geopandas as gpd

# Define paths for baseline and scenario
lulc_2017_path = r"D:\Shared drives\NatCapTEEMs\Projects\WB_MANAGE_Project\ES\LC_SEALS\withFullBoundary\lulc_esa_seals7_2017_full_IND_linear.tif"
lulc_2022_path = r"D:\Shared drives\NatCapTEEMs\Projects\WB_MANAGE_Project\ES\LC_SEALS\withFullBoundary\lulc_esa_seals7_ssp2_rcp45_luh2-message_bau_shift_2022_2022_full_IND_linear.tif"

poll_suff_2017_path = r"D:\Shared drives\NatCapTEEMs\Projects\WB_MANAGE_Project\Shocks\pollination\workspace_poll_suff\seals_ori\churn\poll_suff_hab_ag_coverage_rasters\poll_suff_ag_coverage_prop_10s_lulc_esa_seals7_2017.tif"
poll_suff_2022_path = r"D:\Shared drives\NatCapTEEMs\Projects\WB_MANAGE_Project\Shocks\pollination\workspace_poll_suff\seals_ori\churn\poll_suff_ag_coverage_rasters\ppoll_suff_ag_coverage_prop_10s_lulc_esa_seals7_ssp2_rcp45_luh2-message_bau_shift_2022_2022.tif"

# Crop dependencies paths
crop_prices_dir = r"D:\Shared drives\NatCapTEEMs\Files\base_data\crops\earthstat\crop_prices"
crop_data_dir = r"D:\Shared drives\NatCapTEEMs\Files\base_data\crops\earthstat\crop_production"
pollination_dependence_spreadsheet_input_path = r"D:\Shared drives\NatCapTEEMs\Files\base_data\pollination\rspb20141799supp3.xls"

# Create output paths
output_dir = r"D:\Shared drives\NatCapTEEMs\Projects\WB_MANAGE_Project\Shocks\pollination\outputs"
crop_value_difference_path = os.path.join(output_dir, 'crop_value_difference_from_2017_to_2022.tif')
crop_value_pollinator_adjusted_output_path = os.path.join(output_dir, 'crop_value_pollinator_adjusted_2022.tif')

# Threshold for sufficient pollination
sufficient_pollination_threshold = 0.3

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
L = logging.getLogger()

# Step 1: Load crop dependence data
df_dependence = pd.read_excel(pollination_dependence_spreadsheet_input_path, sheet_name='Crop nutrient content')
crop_names = list(df_dependence['Crop map file name'])[:-3]  # Removing the last 3 crops that don't have matching production data
pollination_dependence = list(df_dependence['poll.dep'])

# Step 2: Initialize arrays for calculations
ha_shape = hb.get_shape_from_dataset_path(lulc_2017_path)
crop_value_baseline = np.zeros(ha_shape)
crop_value_no_pollination = np.zeros(ha_shape)

# Step 3: Calculate baseline crop value and no-pollination value
for c, crop_name in enumerate(crop_names):
    L.info(f'Calculating crop value for {crop_name} with pollination dependence {pollination_dependence[c]}')
    
    # Load crop prices and production data
    crop_price_path = os.path.join(crop_prices_dir, f'{crop_name}_prices_per_ton.tif')
    crop_price = hb.as_array(crop_price_path)
    crop_price = np.where(crop_price > 0, crop_price, 0.0)
    
    crop_yield_path = os.path.join(crop_data_dir, f'{crop_name}_HarvAreaYield_Geotiff', f'{crop_name}_Production.tif')
    crop_yield = hb.as_array(crop_yield_path)
    crop_yield = np.where(crop_yield > 0, crop_yield, 0.0)

    # Calculate value
    crop_value_baseline += (crop_yield * crop_price)
    crop_value_no_pollination += (crop_yield * crop_price) * (1 - float(pollination_dependence[c]))

# Step 4: Calculate maximum loss due to pollination
crop_value_max_lost = crop_value_baseline - crop_value_no_pollination

# Step 5: Save the baseline crop values
hb.save_array_as_geotiff(crop_value_baseline, os.path.join(output_dir, 'crop_value_baseline_2017.tif'), lulc_2017_path, ndv=-9999, data_type=6)
hb.save_array_as_geotiff(crop_value_max_lost, os.path.join(output_dir, 'crop_value_max_lost_2017.tif'), lulc_2017_path, ndv=-9999, data_type=6)

# Step 6: Resample rasters to match LULC
hb.resample_to_match(poll_suff_2022_path, lulc_2022_path, poll_suff_2022_path, ndv=-9999., output_data_type=6)

# Step 7: Load LULC and Pollination Sufficiency data for 2017 and 2022
lulc_2017 = hb.as_array(lulc_2017_path)
lulc_2022 = hb.as_array(lulc_2022_path)
poll_suff_2017 = hb.as_array(poll_suff_2017_path)
poll_suff_2022 = hb.as_array(poll_suff_2022_path)

# Step 8: Calculate crop value adjusted for pollination sufficiency in 2022
L.info('Calculating crop value adjusted for pollination sufficiency in 2022')

crop_value_pollinator_adjusted = np.where(
    (crop_value_max_lost > 0) & (poll_suff_2022 < sufficient_pollination_threshold) & (lulc_2022 == 2),
    crop_value_baseline - crop_value_max_lost * (1 - (1 / sufficient_pollination_threshold) * poll_suff_2022),
    np.where((crop_value_max_lost > 0) & (poll_suff_2022 >= sufficient_pollination_threshold) & (lulc_2022 == 2),
             crop_value_baseline, -9999.)
)

# Step 9: Save the crop value adjusted for pollination sufficiency
hb.save_array_as_geotiff(crop_value_pollinator_adjusted, crop_value_pollinator_adjusted_output_path, lulc_2022_path, ndv=-9999, data_type=6)

# Step 10: Calculate difference in crop values between 2017 and 2022
L.info('Calculating difference in crop value between 2017 and 2022')
crop_value_difference = crop_value_pollinator_adjusted - crop_value_baseline

# Step 11: Save the difference in crop values
hb.save_array_as_geotiff(crop_value_difference, crop_value_difference_path, lulc_2022_path, ndv=-9999, data_type=6)

# Step 12: Calculate shock value
L.info('Calculating shock value')
shock = np.sum(crop_value_difference) / np.sum(crop_value_baseline)
L.info(f'Shock value: {shock}')
