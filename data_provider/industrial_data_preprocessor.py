import os
import logging
import argparse
import pandas as pd
import numpy as np
import pyreadr


class IndustrialPreprocessor:
    def __init__(self):
        self.TEP_VAR_MAPPING = {
            'xmv_1': 'D_Feed_Flow_Valve',
            'xmv_2': 'E_Feed_Flow_Valve',
            'xmv_3': 'A_Feed_Flow_Valve',
            'xmv_4': 'Total_Feed_Flow_Valve',
            'xmv_10': 'Reactor_Cooling_Valve',
            'xmeas_1': 'A_Feed_Flow_Rate',
            'xmeas_6': 'Reactor_Feed_Rate',
            'xmeas_8': 'Reactor_Level',
            'xmeas_9': 'Reactor_Temp',
            'xmeas_7': 'Reactor_Pressure'  # Target Variable
        }
        self.TEP_ORDER = list(self.TEP_VAR_MAPPING.values())

    def process_tep(self, input_path, output_path):
        result = pyreadr.read_r(input_path)       
        raw_df = result[list(result.keys())[0]]      
        df_processed = raw_df[list(self.TEP_VAR_MAPPING.keys())].rename(columns=self.TEP_VAR_MAPPING)       
        df_processed = df_processed[self.TEP_ORDER]
        timestamps = pd.date_range(
            start='2020-01-01', 
            periods=len(df_processed), 
            freq='3min'
        )
        df_processed.insert(0, 'date', timestamps)

        df_processed.to_csv(output_path, index=False)
        print(f"TEP processing complete. Saved to {output_path}. Shape: {df_processed.shape}")

    def process_sdwpf(self, input_path, output_path, turbine_id=1):       
        df = pd.read_csv(input_path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        schema_variants = {
            'TurbID': ['TurbID', 'turbid', 'Turbine_ID'],
            'Day': ['Day', 'day'],
            'Tmstmp': ['Tmstamp', 'Tmstmp', 'tmstmp', 'tmstamp', 'Timestamp'],
            'Wspd': ['Wspd', 'wspd', 'Wind_Speed'],
            'Wdir': ['Wdir', 'wdir', 'Wind_Direction'],
            'Etmp': ['Etmp', 'etmp', 'Environment_Temp'],
            'Itmp': ['Itmp', 'itmp', 'Internal_Temp'],
            'Ndir': ['Ndir', 'ndir', 'Nacelle_Direction'],
            'Pab1': ['Pab1', 'pab1', 'Pitch_Angle'],
            'Patv': ['Patv', 'patv', 'Active_Power']
        }

        # Apply mapping
        rename_map = {}
        for canonical, variants in schema_variants.items():
            for variant in variants:
                if variant in df.columns:
                    rename_map[variant] = canonical
                    break
        
        df = df.rename(columns=rename_map)

        # Validate mandatory columns
        required = ['TurbID', 'Day', 'Tmstmp', 'Patv']
        missing = [col for col in required if col not in df.columns]

        df_t = df[df['TurbID'] == turbine_id].copy()

        # Vectorized DateTime Construction
        base_date = pd.to_datetime('2020-01-01')
        
        days_offset = pd.to_timedelta(df_t['Day'] - 1, unit='D')
        time_series = df_t['Tmstmp'].astype(str)
        time_offset = pd.to_timedelta(time_series.apply(lambda x: x if len(x.split(':')) > 2 else x + ':00'))
        df_t['date'] = base_date + days_offset + time_offset
        df_t['Patv'] = df_t['Patv'].clip(lower=0)

        df_t = df_t.ffill().fillna(0)

        selected_features = ['date', 'Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Patv']
        final_cols = [c for c in selected_features if c in df_t.columns]
        final_df = df_t[final_cols]

        final_df.to_csv(output_path, index=False)
        print(f"SDWPF processing complete. Saved to {output_path}. Shape: {final_df.shape}")


def main():
    tep_input="TEP_FaultFree_Training.RData"
    sdwpf_input="wtbdata_245days.csv"
    processor = IndustrialPreprocessor()
    processor.process_tep(tep_input, "tep_reactor_pressure_target.csv")
    processor.process_sdwpf(sdwpf_input, "sdwpf_turbine_1_processed.csv")

if __name__ == "__main__":
    main()
