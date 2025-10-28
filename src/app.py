import traceback  # For detailed error printing
import urllib.parse  # For URL encoding in details link

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State, no_update
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sqlalchemy import create_engine, text


# ==============================================================================
# 1. CORE MBAE ANALYSIS FUNCTIONS
# ==============================================================================

def run_historical_analysis(corridor_filter, disruption_filter):
    #Docstring explaining how the function works + outputs
    """
    Performs the main historical analysis.
    This function connects to the database, fetches performance data,
    identifies all disruption cycles, and then calculates the average
    Time-to-Recover (TTR) and Adaptation (ADP) durations.

    It generates a smoothed "archetypal" curve from all cycles
    and calculates the baseline Resilience Loss (Re) from that curve.

    Returns a tuple with:
    (fitted_cycles_data, final_archetype_curve, avg_ttr_days,
     avg_baseline_re, cycle_count, min_perf, avg_adp_hours)
    """

    # Initialize return values
    fitted_cycles_data = []  # Store individual fitted cycle data {x_hours, y_perf}
    final_archetype_curve = None
    avg_ttr_days = 0
    avg_baseline_re = 0  # Calculated from the final archetype
    cycle_count = 0
    min_perf = 0
    avg_adp_hours = 0

    try:
        # Database connection parameters
        db_params = {
            "user": "postgres", "password": "Sipho$e2", "host": "localhost",
            "port": "5432", "database": "tfr_resilience_db4"
        }
        engine = create_engine(
            f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}")

        # This SQL query aggregates train logs into 4-hour blocks, calculates performance,
        # and flags which blocks were part of the selected disruption type.
        sql_query = text("""
                         WITH DisruptionEvents AS (SELECT disruption_id
                                                   FROM Disruptions d
                                                            JOIN Corridor c ON d.corridor_id = c.corridor_id
                                                   WHERE d.disruption_type = :disruption_type
                                                     AND c.corridor_name = :corridor_name)
                         SELECT to_timestamp(floor(extract(epoch from tl.departure_time AT TIME ZONE 'UTC') / 14400) *
                                             14400) AT TIME ZONE 'UTC'                 AS time_block,
                                COUNT(*)                                               AS total_trips,
                                SUM(CASE WHEN tl.status = 'Arrived' THEN 1 ELSE 0 END) AS successful_trips,
                                MAX(CASE
                                        WHEN tl.disruption_id IN (SELECT disruption_id FROM DisruptionEvents) THEN 1
                                        ELSE 0 END)                                    as is_selected_disruption_block
                         FROM TrainLog tl
                                  JOIN Corridor c ON tl.corridor_id = c.corridor_id
                         WHERE c.corridor_name = :corridor_name
                         GROUP BY time_block
                         ORDER BY time_block;
                         """)
        df_4hr = pd.read_sql(sql_query, engine,
                             params={'corridor_name': corridor_filter, 'disruption_type': disruption_filter})

        if df_4hr.empty:
            print(f"No data found for {corridor_filter}")
            return [], None, 0, 0, 0, 0, 0

        # Calculate performance as a ratio, filling any NaN values with 1.0 (full performance)
        df_4hr['performance'] = ((df_4hr['successful_trips'] / df_4hr['total_trips']).fillna(1.0)).clip(upper=1.0)
        df_4hr['time_block'] = pd.to_datetime(df_4hr['time_block'])

        # --- Cycle Isolation ---
        PERFORMANCE_THRESHOLD = 0.90  # Defines the start of a disruption
        BASELINE_RECOVERY_THRESHOLD = 0.93  # Defines the end of a disruption
        PADDING = 4  # Number of data points to add before/after the cycle for better curve fitting
        in_disruption = False
        disruption_cycles_raw_padded = []
        actual_ttrs_days = []
        adaptation_durations_hours = []
        cycle_start_index = -1

        # Iterate through the performance data to identify distinct disruption cycles.
        for i in range(len(df_4hr)):
            row = df_4hr.iloc[i]  # Get the row using the integer position
            is_below_threshold = row['performance'] < PERFORMANCE_THRESHOLD
            # A disruption cycle starts when performance drops below the threshold
            # and is flagged as the correct disruption type.
            if not in_disruption and is_below_threshold and row['is_selected_disruption_block'] == 1:
                in_disruption = True
                cycle_start_index = i

            # A cycle ends when performance goes back above the threshold.
            elif in_disruption and not is_below_threshold:
                in_disruption = False
                cycle_end_index = i
                if cycle_start_index != -1:
                    # Get the unpadded disruption cycle data
                    unpadded_cycle_df = df_4hr.iloc[cycle_start_index:cycle_end_index]

                    if not unpadded_cycle_df.empty and unpadded_cycle_df['is_selected_disruption_block'].any():
                        # Find the full recovery point
                        recovery_end_index = cycle_end_index
                        while recovery_end_index < len(df_4hr) - 1 and df_4hr.iloc[recovery_end_index][
                            'performance'] < BASELINE_RECOVERY_THRESHOLD:
                            recovery_end_index += 1

                        end_slice = min(len(df_4hr), recovery_end_index + PADDING)

                        # Calculate the actual Time-to-Recover (TTR)
                        actual_start_time = df_4hr.iloc[cycle_start_index]['time_block']
                        actual_end_time = df_4hr.iloc[recovery_end_index]['time_block']
                        ttr_hours = (actual_end_time - actual_start_time).total_seconds() / 3600
                        if ttr_hours > 4: #Calculate for disruptions only
                            actual_ttrs_days.append(ttr_hours / 24)

                        # Add padding to the start and end for curve fitting to capture the full disruption cycle
                        start_slice = max(0, cycle_start_index - PADDING)
                        cycle_to_add = df_4hr.iloc[start_slice:end_slice].copy()
                        disruption_cycles_raw_padded.append(cycle_to_add)

                        # Calculate Adaptation Duration
                        y_perf_temp = unpadded_cycle_df['performance']
                        min_perf_value_temp = y_perf_temp.min()
                        trough_threshold_temp = min_perf_value_temp + 0.10
                        trough_points_temp = y_perf_temp[y_perf_temp <= trough_threshold_temp]

                        if not trough_points_temp.empty:
                            try:
                                # Find the most common performance value in the trough points (adaptation phase)
                                true_trough_level_temp = trough_points_temp.mode()[0]
                                adaptation_points_indices_temp = y_perf_temp[
                                    y_perf_temp == true_trough_level_temp].index

                                if len(adaptation_points_indices_temp) > 0:
                                    # Calculate duration of this adaptation phase
                                    adaptation_start_time_temp = df_4hr.loc[adaptation_points_indices_temp[0]][
                                        'time_block']
                                    adaptation_end_time_temp = df_4hr.loc[adaptation_points_indices_temp[-1]][
                                        'time_block']
                                    adp_duration_h = max((
                                                                 adaptation_end_time_temp - adaptation_start_time_temp).total_seconds() / 3600.0,
                                                         4.0)  # Ensure a min duration of 4 hours
                                    adaptation_durations_hours.append(adp_duration_h)
                            except IndexError:
                                pass  #Ignore if mode calculation fails
                cycle_start_index = -1

        #This handles the case where a disruption is still ongoing when the data ends
        if in_disruption and cycle_start_index != -1:
            unpadded_cycle_df = df_4hr.iloc[cycle_start_index:]
            if not unpadded_cycle_df.empty and unpadded_cycle_df['is_selected_disruption_block'].any():
                recovery_end_index = len(df_4hr) - 1
                end_slice = len(df_4hr)

                #Calculate TTR for this partial cycle
                actual_start_time = df_4hr.iloc[cycle_start_index]['time_block']
                actual_end_time = df_4hr.iloc[recovery_end_index]['time_block']
                ttr_hours = (actual_end_time - actual_start_time).total_seconds() / 3600
                if ttr_hours > 4:
                    actual_ttrs_days.append(ttr_hours / 24)

                start_slice = max(0, cycle_start_index - PADDING)
                disruption_cycles_raw_padded.append(df_4hr.iloc[start_slice:end_slice].copy())

                #Calculate Adaptation phase for this partial cycle
                y_perf_temp = unpadded_cycle_df['performance']
                min_perf_value_temp = y_perf_temp.min()
                trough_threshold_temp = min_perf_value_temp + 0.10
                trough_points_temp = y_perf_temp[y_perf_temp <= trough_threshold_temp]

                if not trough_points_temp.empty:
                    try:
                        true_trough_level_temp = trough_points_temp.mode()[0]
                        adaptation_points_indices_temp = y_perf_temp[y_perf_temp == true_trough_level_temp].index
                        if len(adaptation_points_indices_temp) > 0:
                            adaptation_start_time_temp = df_4hr.loc[adaptation_points_indices_temp[0]]['time_block']
                            adaptation_end_time_temp = df_4hr.loc[adaptation_points_indices_temp[-1]]['time_block']
                            adp_duration_h = max(
                                (adaptation_end_time_temp - adaptation_start_time_temp).total_seconds() / 3600.0, 4.0)
                            adaptation_durations_hours.append(adp_duration_h)
                    except IndexError:
                        pass  #Ignore if mode calculation fails

        #Calculate the final averages
        avg_ttr_days = np.mean(actual_ttrs_days) if actual_ttrs_days else 0
        avg_adp_hours = np.mean(adaptation_durations_hours) if adaptation_durations_hours else 0
        cycle_count = len(disruption_cycles_raw_padded)
        print(
            f"[Core Logic] Identified {cycle_count} cycles. Avg TTR: {avg_ttr_days:.2f} days. Avg Adp: {avg_adp_hours:.2f} hrs.")

        if cycle_count == 0:
            print(f"No disruption cycles isolated.")
            return [], None, 0, 0, 0, 0, 0

        #Fitting Logic
        #Process each isolated cycle
        fitted_cycles_data = []  #Reset list to store results
        for cycle_df in disruption_cycles_raw_padded:
            if len(cycle_df) < 5:
                continue  #Not enough data to fit a curve

            cycle_df['time_block'] = pd.to_datetime(cycle_df['time_block'])
            #Convert time to relative hours from the start of the padded cycle
            x_time_hours_relative = (cycle_df['time_block'] - cycle_df['time_block'].iloc[
                0]).dt.total_seconds() / 3600.0
            y_perf = cycle_df['performance']

            #Skip minor dips that are too close to 100% performance
            min_perf_value = y_perf.min()
            if min_perf_value > 0.98:
                continue

            #Find the adaptation phase again for splitting
            trough_threshold = min_perf_value + 0.10
            trough_points = y_perf[y_perf <= trough_threshold]
            if trough_points.empty:
                continue

            try:
                true_trough_level = trough_points.mode()[0]
                adaptation_points_indices = y_perf[y_perf == true_trough_level].index
                if len(adaptation_points_indices) == 0:
                    continue

                adaptation_start_idx = adaptation_points_indices[0]
                adaptation_end_idx = adaptation_points_indices[-1]

                #Get integer-based locations for splitting
                relative_start_loc = cycle_df.index.get_loc(adaptation_start_idx)
                relative_end_loc = cycle_df.index.get_loc(adaptation_end_idx)

                if relative_start_loc >= relative_end_loc:
                    if relative_start_loc == relative_end_loc:
                        relative_end_loc = relative_start_loc + 1  # Ensure at least one point in adaptation
                    else:
                        continue  #Invalid split
            except (IndexError, KeyError) as e:
                continue  #Failed to find adaptation points

            #Split the cycle into 3 phases: Absorption, Adaptation, Recovery
            absorption_df = cycle_df.iloc[:relative_start_loc + 1]
            adaptation_df = cycle_df.iloc[relative_start_loc:min(relative_end_loc + 1, len(cycle_df))]
            recovery_df = cycle_df.iloc[min(relative_end_loc + 1, len(cycle_df)):]

            #Ensure enough data in each phase to fit a model
            if len(absorption_df) < 2 or len(adaptation_df) < 1 or len(recovery_df) < 2:
                continue

            try:
                #Prepare X values (relative hours) for each phase
                X_abs = x_time_hours_relative.iloc[:relative_start_loc + 1].values.reshape(-1, 1)
                X_adp = x_time_hours_relative.iloc[
                    relative_start_loc:min(relative_end_loc + 1, len(cycle_df))].values.reshape(-1, 1)
                X_rec = x_time_hours_relative.iloc[min(relative_end_loc + 1, len(cycle_df)):].values.reshape(-1, 1)

                if X_abs.size == 0 or X_adp.size == 0 or X_rec.size == 0:
                    continue

                #Fit polynomial models to each phase
                abs_poly = PolynomialFeatures(degree=2)
                X_abs_poly = abs_poly.fit_transform(X_abs)
                abs_model = LinearRegression().fit(X_abs_poly, absorption_df['performance'].values)

                adp_poly = PolynomialFeatures(degree=1 if len(adaptation_df) > 1 else 0)  #Linear for the adapt phase
                X_adp_poly = adp_poly.fit_transform(X_adp)
                adp_model = LinearRegression().fit(X_adp_poly, adaptation_df['performance'].values)

                rec_poly = PolynomialFeatures(degree=2)
                X_rec_poly = rec_poly.fit_transform(X_rec)
                rec_model = LinearRegression().fit(X_rec_poly, recovery_df['performance'].values)

                #Generate smooth curves from the models
                x_smooth_abs = np.linspace(X_abs.min(), X_abs.max(), 50).reshape(-1, 1)
                y_smooth_abs = abs_model.predict(abs_poly.transform(x_smooth_abs))

                x_smooth_adp_rel = np.linspace(X_adp.min(), X_adp.max(), 50).reshape(-1, 1)
                y_smooth_adp = adp_model.predict(adp_poly.transform(x_smooth_adp_rel))

                x_smooth_rec_rel = np.linspace(X_rec.min(), X_rec.max(), 100).reshape(-1, 1)
                y_smooth_rec = rec_model.predict(rec_poly.transform(x_smooth_rec_rel))

                #Stitch the curves together at their connection points to avoid gaps
                if len(y_smooth_abs) > 0 and len(y_smooth_adp) > 0 and x_smooth_abs[-1, 0] >= x_smooth_adp_rel[0, 0]:
                    y_smooth_abs[-1] = y_smooth_adp[0]
                if len(y_smooth_adp) > 0 and len(y_smooth_rec) > 0 and x_smooth_adp_rel[-1, 0] >= x_smooth_rec_rel[
                    0, 0]:
                    y_smooth_adp[-1] = y_smooth_rec[0]

                #Combine all 3 fitted curves into one
                combined_x = np.concatenate(
                    [x_smooth_abs.flatten(), x_smooth_adp_rel.flatten(), x_smooth_rec_rel.flatten()])
                combined_y = np.concatenate([y_smooth_abs.flatten(), y_smooth_adp.flatten(), y_smooth_rec.flatten()])

                #Ensure x-values are unique and sorted
                unique_indices = np.unique(combined_x, return_index=True)[1]
                combined_x = combined_x[unique_indices]
                combined_y = combined_y[unique_indices]
                sort_order = np.argsort(combined_x)
                combined_x = combined_x[sort_order]
                combined_y = combined_y[sort_order]

                fitted_cycles_data.append({"x": combined_x, "y": combined_y})
            except ValueError as fit_error:
                continue  #Ignore if a model fails to fit

        if not fitted_cycles_data:
            print(f"No cycles could be successfully fitted.")
            return [], None, avg_ttr_days, 0, cycle_count, 0, avg_adp_hours

        #Archetype Generation
        #Normalize all fitted curves to a 0-100 timeline to compare them
        normalized_x = np.linspace(0, 100, 101)
        all_normalized_curves = []
        for cycle in fitted_cycles_data:
            original_x, original_y = cycle['x'], cycle['y']
            duration = original_x.max() - original_x.min()
            if duration > 0:
                all_normalized_curves.append(
                    interp1d((original_x - original_x.min()) / duration * 100, original_y, bounds_error=False,
                             fill_value="extrapolate")(normalized_x))

        if not all_normalized_curves:
            return fitted_cycles_data, None, avg_ttr_days, 0, cycle_count, 0, avg_adp_hours

        #Average all normalized curves to create the archetype
        archetypal_curve_y = np.array(all_normalized_curves).mean(axis=0)

        #Apply a final smoothing polynomial to the averaged archetype
        final_poly = PolynomialFeatures(degree=7)
        X_final_poly = final_poly.fit_transform(normalized_x.reshape(-1, 1))
        final_model = LinearRegression().fit(X_final_poly, archetypal_curve_y)
        final_smooth_y = final_model.predict(X_final_poly)
        final_smooth_y = np.clip(final_smooth_y, 0, 1.0)  #Ensure performance is between 0 and 1

        final_archetype_curve = {"x": normalized_x, "y": final_smooth_y}

        #Calculate the baseline Resilience Loss (Re) from our final archetype curve.
        #Scale the normalized (0-100) x-axis to the average TTR in hours.
        base_ttr_hours_for_re = avg_ttr_days * 24 if avg_ttr_days > 0 else 24
        baseline_x_hours_for_re = final_archetype_curve['x'] * (base_ttr_hours_for_re / 100)
        archetype_y_for_re = np.clip(final_archetype_curve['y'], 0, 1.0)

        #Calculate Re (Resilience Loss) as the area under the (1 - performance) curve
        avg_baseline_re = np.trapezoid(1.0 - archetype_y_for_re, baseline_x_hours_for_re)

        min_perf = final_archetype_curve['y'].min() if final_archetype_curve else 0

        #Return all results
        return fitted_cycles_data, final_archetype_curve, avg_ttr_days, avg_baseline_re, cycle_count, min_perf, avg_adp_hours

    except Exception as e:
        print(f"ERROR in run_historical_analysis: {e}")
        traceback.print_exc()
        return [], None, 0, 0, 0, 0, 0


def analyze_investment_scenario(final_archetype_curve, avg_ttr_days, avg_adp_hours, financial_params,
                                investment_params, avg_baseline_re):

    if final_archetype_curve is None:
        return {"baseline_curve": {"x": [], "y": [], "Re": 0, "TDC": 0}, "scenario_curve": {"x": [], "y": [], "Re": 0},
                "business_case": {"Scenarios": investment_params['name'], "Resilience loss (Re)": 0,
                                  "TDC (R)": investment_params.get('implementation_cost', 0), "ROI": -100.0,
                                  "Payback Years": float('inf'), "BCR": 0, "Benefit": 0,
                                  "Cost": investment_params.get('implementation_cost', 0)},
                "inputs_used": investment_params}

    BASELINE_PERFORMANCE = 1.0

    #Use the avg_baseline_re passed in, which was calculated from the archetype curve.
    archetype_baseline_re = avg_baseline_re
    archetype_baseline_tdc = financial_params['Cu'] * archetype_baseline_re  #TDC per event based on archetype baseline

    #Process all investment parameters from the user
    perf_loss_reduction = investment_params.get('perf_loss_reduction', 0) / 100.0 if investment_params.get(
        'perf_loss_reduction') is not None else 0.0
    scenario_ci = investment_params.get('implementation_cost', 0) if investment_params.get(
        'implementation_cost') is not None else 0.0
    annual_opex = investment_params.get('annual_opex', 0) if investment_params.get('annual_opex') is not None else 0.0
    duration_years = investment_params.get('duration_years', 1) if investment_params.get(
        'duration_years') is not None else 1.0
    frequency_per_year = investment_params.get('frequency_per_year')  #Get potentially updated frequency

    #Fallback if frequency wasn't passed or was invalid
    if frequency_per_year is None:
        frequency_per_year = financial_params.get('frequency_per_year', 0)

    #Process time-related parameters
    new_ttr_days_input = investment_params.get('new_ttr_days', None)
    base_ttr_hours = avg_ttr_days * 24 if avg_ttr_days > 0 else 24

    new_adp_days_input = investment_params.get('new_adp_days', None)
    effective_avg_adp_hours = avg_adp_hours if avg_adp_hours > 0 else 4

    #Use the baseline TTR/ADP if the user didn't provide a new value
    user_ttr_hours = base_ttr_hours if new_ttr_days_input is None or new_ttr_days_input <= 0 else new_ttr_days_input * 24
    user_adp_hours = effective_avg_adp_hours if new_adp_days_input is None or new_adp_days_input <= 0 else new_adp_days_input * 24

    #Adjust the final TTR based on any changes to the adaptation duration
    adp_duration_delta_hours = user_adp_hours - effective_avg_adp_hours
    final_new_ttr_hours = max(4, user_ttr_hours + adp_duration_delta_hours)

    #Scenario Curve Modification
    #Apply the investment parameters to the baseline archetype curve
    archetype_x_norm = final_archetype_curve['x']
    archetype_y = np.clip(final_archetype_curve['y'], 0, 1.0)

    #Reduce the performance loss
    modified_y = BASELINE_PERFORMANCE - ((BASELINE_PERFORMANCE - archetype_y) * (1 - perf_loss_reduction))
    modified_y = np.clip(modified_y, 0, 1.0)

    #Rescale the x-axis to the new (scenario) TTR
    modified_x_hours = archetype_x_norm * (final_new_ttr_hours / 100)

    #Calculate the new Resilience Loss (Re) for this scenario
    scenario_re = np.trapezoid(BASELINE_PERFORMANCE - modified_y, modified_x_hours)
    scenario_tdc_event = financial_params['Cu'] * scenario_re  #Scenario TDC per event

    #Financial Calculations
    #Use the archetype's baseline TDC for a consistent comparison.
    savings_per_event = archetype_baseline_tdc - scenario_tdc_event
    annual_savings = (savings_per_event * frequency_per_year) - annual_opex
    calc_duration = max(1, duration_years or 1)

    total_net_profit = (annual_savings * calc_duration) - scenario_ci
    total_roi = (total_net_profit / scenario_ci * 100) if scenario_ci > 0 else float('inf')
    payback_years = (scenario_ci / annual_savings) if annual_savings > 0 else float('inf')
    benefit_value = savings_per_event * frequency_per_year  #Annual gross benefit
    bcr = benefit_value / scenario_ci if scenario_ci > 0 else float('inf')  #Annual gross benefit / CAPEX

    #Baseline curve for visualisation scaled to baseline TTR
    baseline_x_hours_visual = archetype_x_norm * (base_ttr_hours / 100)

    return {
        #Pass the archetype baseline Re and TDC for consistency
        "baseline_curve": {"x": baseline_x_hours_visual, "y": archetype_y, "Re": archetype_baseline_re,
                           "TDC": archetype_baseline_tdc},
        "scenario_curve": {"x": modified_x_hours, "y": modified_y, "Re": scenario_re},
        "business_case": {
            "Scenarios": investment_params['name'],
            "Resilience loss (Re)": scenario_re,
            "TDC (R)": scenario_tdc_event,  #Scenario TDC per event
            "ROI": total_roi,
            "Payback Years": payback_years,
            "BCR": bcr,
            "Benefit": benefit_value,  #Annual Gross Benefit
            "Cost": scenario_ci},
        "inputs_used": investment_params}


# ==============================================================================
# 2. APP INITIALIZATION & LAYOUT DEFINITIONS
# ==============================================================================

#Global financial inputs
financial_inputs = {"Cu": 90138.81, "frequency_per_year": 12}

#Color definitions for the app's theme
COLOR_PRIMARY_GREEN = '#2E8B57'
COLOR_LIGHT_GREEN_F = '#90EE90'
COLOR_ACCENT_RED = '#DC3545'
COLOR_CARD_BACKGROUND = '#D9F2D0'
COLOR_INPUT_BACKGROUND = 'rgba(46, 139, 87, 0.15)'  #Darker transparent green for inputs
COLOR_METRIC_TEXT = '#0D3512'  #Dark green for metric labels
COLOR_BORDER_LIGHT = '#E9ECEF'  #Light grey border for cards
COLOR_CHART_FILL_GREEN = 'rgba(46, 139, 87, 0.3)'  #Semi-transparent green for chart fill
COLOR_CHART_LINE_DARKGREEN = '#0D3512'  #Dark green for chart line
COLOR_GRAPH_BASELINE_LINE = COLOR_CHART_LINE_DARKGREEN
COLOR_GRAPH_OPTIMISTIC = 'green'
COLOR_GRAPH_LIKELY = 'orange'
COLOR_GRAPH_PESSIMISTIC = 'red'
COLOR_PAGE_BACKGROUND = '#F2F2F2'  #Light grey page background
COLOR_TABLE_STRIPE = '#E9ECEF'  #Light grey for table stripe

#Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE], suppress_callback_exceptions=True)
server = app.server

#Custom CSS for styling input placeholders and tables
custom_css = """
body {{ background-color: {COLOR_PAGE_BACKGROUND}; }} /* Apply background to body */
.custom-input::placeholder {{
  color: {COLOR_METRIC_TEXT} !important;
  opacity: 0.7 !important; /* Optional: Adjust opacity */
}}
/* Ensure dropdowns also have the desired background */
.Select-control {{
    background-color: {COLOR_INPUT_BACKGROUND} !important;
}}
.Select--single > .Select-control .Select-value, .Select-placeholder {{
     color: {COLOR_METRIC_TEXT} !important; /* Style dropdown text */
}}
/* Table Styling */
.table-striped > tbody > tr:nth-of-type(odd) > * {{
    background-color: {COLOR_CARD_BACKGROUND} !important; /* Green for odd rows */
    color: {COLOR_METRIC_TEXT} !important; /* Dark text on green */
}}
.table-striped > tbody > tr:nth-of-type(even) > * {{
    background-color: {COLOR_TABLE_STRIPE} !important; /* Grey for even rows */
    color: black !important; /* Black text on grey */
}}
.table > :not(caption) > * > * {{ /* Ensure borders are visible */
    border-bottom-width: 1px;
    border-color: #dee2e6; /* Default bootstrap border color */
}}
""".format(
    COLOR_PAGE_BACKGROUND=COLOR_PAGE_BACKGROUND,
    COLOR_METRIC_TEXT=COLOR_METRIC_TEXT,
    COLOR_INPUT_BACKGROUND=COLOR_INPUT_BACKGROUND,
    COLOR_CARD_BACKGROUND=COLOR_CARD_BACKGROUND,
    COLOR_TABLE_STRIPE=COLOR_TABLE_STRIPE
)

#Inject the custom CSS into the app's HTML index string
app.index_string = f'''
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>
            {custom_css}
        </style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
'''

#Options for the main filter dropdowns
corridor_options = ['North Corridor']
disruption_options = ['Cable Theft', 'Track Failure']

# --- App Layouts ---

sidebar = html.Div(
    [
        html.H3(  # Smaller than H2 for a cleaner look
            ["Transnet ",
             html.Span("F", style={'color': COLOR_LIGHT_GREEN_F, 'fontWeight': 'bold'}),  # Lighter Green F
             html.Span("R", style={'color': COLOR_ACCENT_RED, 'fontWeight': 'bold'})],  # Red R
            className="text-center",
            style={'color': 'white', 'marginBottom': '0'}
        ),
        html.Hr(style={'borderColor': 'white', 'marginTop': '0.5rem'}),
        dbc.Nav([], vertical=True, pills=True, id="nav-links"),
        html.Div([
            dbc.NavLink("Information", href="/info", style={'color': 'white'}),
            dbc.NavLink("Log out", href="/logout", style={'color': 'white'})
        ], style={'position': 'absolute', 'bottom': '2rem'})
    ],
    style={
        'position': 'fixed', 'top': 0, 'left': 0, 'bottom': 0, 'width': '15rem',
        'padding': '2rem 1rem', 'backgroundColor': COLOR_PRIMARY_GREEN
    }
)


def make_input_group(label, mode_id):
    """A helper function to create a Min/Mode/Max input card."""
    #Apply custom-input class and background style to dbc.Input
    input_style = {'backgroundColor': COLOR_INPUT_BACKGROUND}
    return dbc.Card([
        dbc.CardHeader(label),
        dbc.CardBody(dbc.Row([
            dbc.Col(dbc.Input(id=f"{mode_id}-min", type="number", placeholder="Min", className="custom-input",
                              style=input_style)),
            dbc.Col(
                dbc.Input(id=mode_id, type="number", placeholder="Mode", className="custom-input", style=input_style)),
            dbc.Col(dbc.Input(id=f"{mode_id}-max", type="number", placeholder="Max", className="custom-input",
                              style=input_style))
        ]))
    ], className="mb-2 shadow", style={'backgroundColor': COLOR_CARD_BACKGROUND, 'borderColor': COLOR_BORDER_LIGHT})


#This is the main layout for the "Scenario Analysis" page
inputs_page_layout = html.Div([
    dbc.Row([
        dbc.Col(html.H1("Resilience Scenario Inputs", style={'color': 'black'}), width="auto"),
        dbc.Col(dcc.Dropdown(id='corridor-filter', options=corridor_options, value=corridor_options[0], clearable=False,
                             className="custom-input"), className="ms-auto", width=3),
        dbc.Col(dcc.Dropdown(id='disruption-filter', options=disruption_options, value=disruption_options[0],
                             clearable=False, className="custom-input"), width=3)
    ], align="center", className="mb-4"),
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader(id='baseline-curve-title'),
            dbc.CardBody(dcc.Graph(id='baseline-curve-graph'))
        ], style={'backgroundColor': COLOR_CARD_BACKGROUND, 'borderColor': COLOR_BORDER_LIGHT}, className="shadow"),
            width=7),  # Added shadow and border
        dbc.Col([
            html.H4("Resilience Metrics", className="text-center mb-3"),
            # Metric cards are in a vertical stack
            dbc.Card(id='cu-card', children=dbc.CardBody([
                html.P("Cost of unit Resilience loss (Cu)", style={'color': COLOR_METRIC_TEXT}),
                html.H4(f"R {financial_inputs.get('Cu', 0):,.2f}",
                        style={'color': COLOR_ACCENT_RED, 'fontWeight': 'bold'})
            ]), style={'backgroundColor': COLOR_CARD_BACKGROUND, 'borderColor': COLOR_BORDER_LIGHT},
                     className="shadow mb-3"),  # Added shadow, border, margin
            dbc.Card(id='tdc-card', children=dbc.CardBody([
                html.P("Avg. Disruption Cost (TDC)", style={'color': COLOR_METRIC_TEXT}),
                html.H4("R 0.00", style={'color': COLOR_ACCENT_RED, 'fontWeight': 'bold'}, id='tdc-value-h4')
            ]), style={'backgroundColor': COLOR_CARD_BACKGROUND, 'borderColor': COLOR_BORDER_LIGHT},
                     className="shadow mb-3"),
            dbc.Card(id='ploss-card', children=dbc.CardBody([
                html.P("Avg. Performance loss (α)", style={'color': COLOR_METRIC_TEXT}),
                html.H4("0%", style={'color': COLOR_ACCENT_RED, 'fontWeight': 'bold'}, id='ploss-value-h4')
            ]), style={'backgroundColor': COLOR_CARD_BACKGROUND, 'borderColor': COLOR_BORDER_LIGHT}, className="shadow")
        ], width=5)
    ], className="mb-4"),

    html.H3([
        html.Span("Resilience ", style={'color': COLOR_PRIMARY_GREEN}),
        html.Span("Investment", style={'color': COLOR_ACCENT_RED}), " Scenario ", "Parameters"
    ], className="text-center mb-3"),
    dbc.Row([
        dbc.Col([
            dbc.Row([
                dbc.Col(dbc.Input(id="measure-name-input", placeholder="Measure Name", className="custom-input",
                                  style={'backgroundColor': COLOR_INPUT_BACKGROUND}), width=6),
                dbc.Col(dbc.Input(id="duration-input", placeholder="Effective Duration (No. of years)", type="number",
                                  className="custom-input", style={'backgroundColor': COLOR_INPUT_BACKGROUND}), width=6)
            ], className="mb-3"),
            make_input_group("Fixed Implementation Cost (CAPEX)", "cost-input"),
            make_input_group("Annual Variable Cost (OPEX)", "opex-input"),
        ], width=4),
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    html.H5([html.Span("Absorption", style={'color': COLOR_ACCENT_RED}), " Parameters"],
                            className="text-center"),
                    make_input_group("Reduce Performance Loss By (%)", "ploss-input"),
                    make_input_group("Time to Reach Min Performance", "reach-min-input"),
                    make_input_group("Monthly Disruption Frequency", "freq-input")
                ]),
                dbc.Col([
                    html.H5([html.Span("Adaptation", style={'color': COLOR_ACCENT_RED}), " Parameters"],
                            className="text-center"),
                    make_input_group("Time at Min Performance (Days)", "at-min-input")
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.H5([html.Span("Recovery", style={'color': COLOR_ACCENT_RED}), " Parameters"],
                            className="text-center mt-3"),
                    make_input_group("Time to Recover (TTR) (Days)", "ttr-input")
                ])
            ], className="mt-3")
        ], width=6),
        dbc.Col([
            html.H5("Scenarios Added:", className="mb-2"),
            html.Div(id='scenarios-list', children=[], className="mb-3"),
            dbc.Button("Run Comparison", id="run-btn", color="success", className="w-100 mb-2"),
            dbc.Button("Add Scenario", id="add-btn", color="info", className="w-100 mb-2"),
            dbc.Button("Clear Scenarios", id="clear-btn", color="danger", className="w-100")
        ], width=2, className="d-flex flex-column justify-content-start")
    ], className="mb-4")
])

#Placeholders for the other app pages
report_page_layout = html.Div(id="report-content")
details_page_layout = html.Div(id="details-content")

#This div will hold the content for whatever page the user is on
content = html.Div(id="page-content",
                   style={'marginLeft': '15rem', 'padding': '2rem 1rem', 'backgroundColor': COLOR_PAGE_BACKGROUND})

#The overall app layout: sidebar, URL/data stores, and the page content
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Store(id='scenario-data-store', data=[]),  #Stores results of all added scenarios
    dcc.Store(id='scenario-input-store', data={}),  #Stores the inputs for all added scenarios
    sidebar,
    content
])


# ==============================================================================
# 3. CALLBACKS
# ==============================================================================

@app.callback(
    Output('baseline-curve-graph', 'figure'),
    Output('baseline-curve-title', 'children'),
    Output('tdc-value-h4', 'children'),
    Output('ploss-value-h4', 'children'),
    Input('corridor-filter', 'value'),
    Input('disruption-filter', 'value')
)
def update_profile_callback(corridor, disruption):
    """
    This callback triggers when the user changes the corridor or disruption filters.
    It runs the main historical analysis and updates the baseline curve and metric cards.
    """
    fitted, archetype, avg_ttr, avg_re, count, min_perf, avg_adp_hours = run_historical_analysis(corridor, disruption)

    tdc_value = "N/A"
    ploss_value = "N/A"
    #Default figure if no data is found
    fig = go.Figure().update_layout(
        title_text="No Data Found or No Cycles Isolated", template='plotly_white',
        plot_bgcolor=COLOR_CARD_BACKGROUND, paper_bgcolor=COLOR_CARD_BACKGROUND,
        xaxis_showgrid=False, yaxis_showgrid=False  # Remove gridlines
    )
    title = "N/A"

    if archetype:
        #Scale x-axis to the average TTR in hours
        x_hours_axis = archetype['x'] * (avg_ttr * 24 / 100) if avg_ttr > 0 else archetype['x']
        y_perf = archetype['y']
        y_baseline = np.ones_like(y_perf)  # Array of 1.0s for the top line (max performance)

        fig = go.Figure()
        #Add the 100% baseline trace first
        fig.add_trace(go.Scatter(x=x_hours_axis, y=y_baseline, mode='lines', line=dict(width=0), showlegend=False))
        #Add the performance curve, filling up to the baseline trace
        fig.add_trace(go.Scatter(
            x=x_hours_axis, y=y_perf, fill='tonexty', mode='lines',
            line=dict(color=COLOR_CHART_LINE_DARKGREEN, width=3),
            fillcolor=COLOR_CHART_FILL_GREEN, name='Performance'
        ))

        min_y_range = max(0, min_perf - 0.1 if min_perf is not None else 0.5)
        upper_y_range = 1.05

        fig.update_layout(
            xaxis_title="Time (Hours)",
            yaxis_range=[min_y_range, upper_y_range], yaxis_tickformat='.0%',
            margin=dict(t=30, l=40, r=20, b=40),  # Adjusted top margin
            template='plotly_white',
            plot_bgcolor=COLOR_CARD_BACKGROUND, paper_bgcolor=COLOR_CARD_BACKGROUND,
            showlegend=False,
            xaxis_showgrid=False, yaxis_showgrid=False  # Remove gridlines
        )

        #Add the "Re = ..." annotation in the chart
        center_x_index = len(x_hours_axis) // 2
        annotation_x = x_hours_axis[center_x_index]
        annotation_y = (y_perf[center_x_index] + 1.0) / 2

        fig.add_annotation(
            x=annotation_x, y=annotation_y,
            #Use HTML for multi-line and styling
            text=f"<span style='color:{COLOR_ACCENT_RED}; font-weight:bold; font-size: 14px;'>Re = {avg_re:,.2f}</span><br><span style='color:{COLOR_METRIC_TEXT}; font-size: 12px;'>Resilience Loss Index</span>",
            showarrow=False,
            align="center"
        )

        title = f"{corridor} Performance Under {disruption} Disruption ({count} cycles found)"

        #Use the archetype's Re value to calculate the baseline TDC
        tdc_val_num = avg_re * financial_inputs['Cu'] if avg_re > 0 else 0
        tdc_value = f"R {tdc_val_num:,.0f}"
        perf_loss_num = (1.0 - min_perf) if min_perf is not None else 0
        perf_loss_num = max(0, perf_loss_num)
        ploss_value = f"{perf_loss_num:.0%}"

    return fig, title, tdc_value, ploss_value


@app.callback(
    Output('scenario-data-store', 'data', allow_duplicate=True),
    Output('scenario-input-store', 'data', allow_duplicate=True),
    Output('scenarios-list', 'children'),
    Output('measure-name-input', 'value'),
    Input('add-btn', 'n_clicks'),
    [State('scenario-data-store', 'data'), State('scenario-input-store', 'data'), State('measure-name-input', 'value'),
     State('duration-input', 'value'), State('cost-input-min', 'value'), State('cost-input', 'value'),
     State('cost-input-max', 'value'),
     State('opex-input-min', 'value'), State('opex-input', 'value'), State('opex-input-max', 'value'),
     State('ploss-input-min', 'value'), State('ploss-input', 'value'), State('ploss-input-max', 'value'),
     State('reach-min-input-min', 'value'), State('reach-min-input', 'value'), State('reach-min-input-max', 'value'),
     State('freq-input-min', 'value'), State('freq-input', 'value'), State('freq-input-max', 'value'),
     State('at-min-input-min', 'value'), State('at-min-input', 'value'), State('at-min-input-max', 'value'),
     State('ttr-input-min', 'value'), State('ttr-input', 'value'), State('ttr-input-max', 'value'),
     State('corridor-filter', 'value'), State('disruption-filter', 'value')],
    prevent_initial_call=True
)
def add_scenario_callback(n_clicks, existing_results, existing_inputs, name, duration_years,
                          capex_min, capex_mode, capex_max, opex_min, opex_mode, opex_max,
                          ploss_min, ploss_mode, ploss_max,
                          reach_min_min, reach_min_mode, reach_min_max,
                          freq_min, freq_mode, freq_max,
                          adp_min, adp_mode, adp_max,
                          ttr_min, ttr_mode, ttr_max,
                          corridor, disruption):
    """
    This callback activates when the "Add Scenario" button is clicked.
    It gathers all inputs, runs the analysis, and stores the results.
    """
    if not n_clicks:
        return no_update, no_update, no_update, no_update

    scenario_name = name or f"Scenario {len(existing_results) + 1}"

    #Run the historical analysis to get the baseline archetype for this corridor.
    fitted_data, archetype, avg_ttr, avg_baseline_re_from_archetype, count, _, avg_adp_hours = run_historical_analysis(
        corridor, disruption)
    if archetype is None:
        return no_update, no_update, "Could not run analysis: No baseline data.", no_update

    #Store all inputs (min, mode, max) for the "Detailed Analysis" page
    new_input_data = {scenario_name: {
        "name": scenario_name, "corridor": corridor, "disruption": disruption, "duration_years": duration_years,
        "implementation_cost": {"min": capex_min, "mode": capex_mode, "max": capex_max},
        "annual_opex": {"min": opex_min, "mode": opex_mode, "max": opex_max},
        "perf_loss_reduction": {"min": ploss_min, "mode": ploss_mode, "max": ploss_max},
        "time_to_reach_min": {"min": reach_min_min, "mode": reach_min_mode, "max": reach_min_max},
        "disruption_frequency": {"min": freq_min, "mode": freq_mode, "max": freq_max},
        "new_adp_days": {"min": adp_min, "mode": adp_mode, "max": adp_max},
        "new_ttr_days": {"min": ttr_min, "mode": ttr_mode, "max": ttr_max}}}
    updated_inputs = {**existing_inputs, **new_input_data}

    #Pass the "Mode" values to the investment scenario analysis
    mode_investment_params = {
        "name": scenario_name, "duration_years": duration_years, "implementation_cost": capex_mode or 0,
        "annual_opex": opex_mode or 0, "perf_loss_reduction": ploss_mode or 0,
        "frequency_per_year": freq_mode,  # Pass monthly frequency
        "new_adp_days": adp_mode,
        "new_ttr_days": ttr_mode,
        "corridor": corridor, "disruption": disruption
    }

    #Convert monthly frequency to annual for financial calculations
    if mode_investment_params["frequency_per_year"] is not None:
        try:
            mode_investment_params["frequency_per_year"] = float(mode_investment_params["frequency_per_year"]) * 12
        except (ValueError, TypeError):
            print("Warning: Could not convert monthly frequency to annual. Using default.")
            mode_investment_params["frequency_per_year"] = None  # Let the analysis function use the default

    #Call the analysis function.
    mode_result = analyze_investment_scenario(archetype, avg_ttr, avg_adp_hours, financial_inputs,
                                              mode_investment_params, avg_baseline_re=avg_baseline_re_from_archetype)

    #Add the new result to the list of existing results
    updated_results = existing_results + [mode_result]

    #Update the list of scenario badges on the UI
    scenario_badges = [dbc.Badge(s['business_case']['Scenarios'], color="primary", className="me-1 mb-1") for s in
                       updated_results]

    return updated_results, updated_inputs, scenario_badges, ''  #Clear the measure name input


@app.callback(Output('url', 'pathname', allow_duplicate=True), Input('run-btn', 'n_clicks'), prevent_initial_call=True)
def navigate_to_report(n_clicks):
    """Redirects the user to the /report page when "Run Comparison" is clicked."""
    if n_clicks:
        return '/report'
    return no_update


@app.callback(
    Output('scenario-data-store', 'data', allow_duplicate=True),
    Output('scenario-input-store', 'data', allow_duplicate=True),
    Output('scenarios-list', 'children', allow_duplicate=True),
    Input('clear-btn', 'n_clicks'), prevent_initial_call=True
)
def clear_scenarios_callback():
    """Clears all stored scenario data and inputs."""
    return [], {}, []


@app.callback(Output('report-content', 'children'), Input('scenario-data-store', 'data'))
def update_report_page_callback(stored_data):
    """
    This callback generates the entire Report page.
    It triggers whenever the scenario-data-store is updated (scenarios added or cleared).
    """
    if not stored_data:
        return html.Div("Please add one or more scenarios from the 'Scenario Analysis' page first.")

    try:
        #Get baseline data from the first stored scenario (it's the same for all)
        baseline_re = stored_data[0]['baseline_curve']['Re']
        baseline_tdc = stored_data[0]['baseline_curve']['TDC']
    except (IndexError, KeyError, TypeError):
        return html.Div("Error accessing baseline data from stored scenarios.")

    def create_report_page(results, fin_in, base_re, base_tdc):
        """Helper function to build the report layout."""

        #Performance Curves Chart
        fig_curves = go.Figure()
        #Add the baseline curve
        fig_curves.add_trace(
            go.Scatter(x=results[0]['baseline_curve']['x'], y=results[0]['baseline_curve']['y'], mode='lines',
                       line=dict(color=COLOR_GRAPH_BASELINE_LINE, dash='dash', width=2), name='Baseline'))
        #Add a curve for each scenario
        for res in results:
            fig_curves.add_trace(
                go.Scatter(x=res['scenario_curve']['x'], y=res['scenario_curve']['y'], mode='lines', line=dict(width=3),
                           name=res['business_case']['Scenarios']))

        fig_curves.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),  # Added top margin back for title
            template='plotly_white',
            legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
            yaxis_tickformat='.0%', title="Effects of Potential Resilience Investments on Performance",
            plot_bgcolor=COLOR_CARD_BACKGROUND, paper_bgcolor=COLOR_CARD_BACKGROUND,
            xaxis_showgrid=False, yaxis_showgrid=False  # Remove gridlines
        )

        #Benefit vs Cost & BCR Chart
        bcr_fig = make_subplots(specs=[[{"secondary_y": True}]])
        names = [s['business_case']['Scenarios'] for s in results]
        benefit_values = [s['business_case']['Benefit'] for s in results]
        cost_values = [s['business_case']['Cost'] for s in results]
        bcr_values = [s['business_case']['BCR'] for s in results]

        bcr_fig.add_trace(go.Bar(x=names, y=cost_values, name='Investment Cost (CAPEX)', marker_color=COLOR_ACCENT_RED),
                          secondary_y=False)
        bcr_fig.add_trace(
            go.Bar(x=names, y=benefit_values, name='Annual Benefit Value', marker_color=COLOR_PRIMARY_GREEN),
            secondary_y=False)
        bcr_fig.add_trace(go.Scatter(x=names, y=bcr_values, name='BCR', mode='lines+markers', line=dict(color='black')),
                          secondary_y=True)  #BCR line uses the secondary y-axis

        bcr_fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=40),  #Increased bottom margin for legend
            barmode='group', template='plotly_white',
            title="Benefit vs Cost & BCR for Resilience Scenarios",
            plot_bgcolor=COLOR_CARD_BACKGROUND, paper_bgcolor=COLOR_CARD_BACKGROUND,
            xaxis_showgrid=False, yaxis_showgrid=False,  #Remove gridlines
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
        )
        bcr_fig.update_yaxes(title_text="Rands", secondary_y=False, showgrid=False)
        bcr_fig.update_yaxes(title_text="BCR Value", secondary_y=True, showgrid=False)

        #Ranking Table
        baseline_row = pd.DataFrame(
            [{"Scenarios": "Baseline", "Resilience loss (Re)": base_re, "TDC (R)": base_tdc, "ROI": 0.0,
              "Payback Years": float('inf'), "Details": ""}])
        try:
            results_df = pd.concat([pd.DataFrame([s['business_case']]) for s in results], ignore_index=True)
        except Exception as df_error:
            results_df = pd.DataFrame()

        if not results_df.empty:
            #Add the "View Details" link to each row
            results_df['Details'] = results_df['Scenarios'].apply(
                lambda name: dcc.Link("View Details", href=f"/details/{urllib.parse.quote(name)}"))
            #Combine scenarios with the baseline row and sort by ROI
            ranking_df = pd.concat([results_df, baseline_row], ignore_index=True).sort_values(by='ROI', ascending=False)

            #Format the numbers in the table for display
            for col in ['Resilience loss (Re)', 'TDC (R)', 'ROI', 'Payback Years']:
                if col in ranking_df.columns:
                    if col == 'ROI':
                        ranking_df[col] = ranking_df[col].apply(
                            lambda x: f"{x:,.1f}%" if isinstance(x, (int, float)) and np.isfinite(x) else (
                                "inf" if x == float('inf') else "N/A"))
                    elif col == 'Payback Years':
                        ranking_df[col] = ranking_df[col].apply(
                            lambda x: f"{x:,.1f}" if isinstance(x, (int, float)) and np.isfinite(x) else (
                                "inf" if x == float('inf') else "N/A"))
                    else:
                        ranking_df[col] = ranking_df[col].apply(
                            lambda x: f"R {x:,.0f}" if col == 'TDC (R)' and isinstance(x, (int, float)) else (
                                f"{x:,.2f}" if isinstance(x, (int, float)) else x))
        else:
            #Handle case where there are no results, just show baseline
            ranking_df = baseline_row
            for col in ['Resilience loss (Re)', 'TDC (R)', 'ROI', 'Payback Years']:
                if col in ranking_df.columns:
                    if col == 'Payback Years':
                        ranking_df[col] = "N/A"
                    elif col == 'ROI':
                        ranking_df[col] = f"{ranking_df[col].iloc[0]:,.1f}%"
                    elif col == 'TDC (R)':
                        ranking_df[col] = f"R {ranking_df[col].iloc[0]:,.0f}"
                    else:
                        ranking_df[col] = f"{ranking_df[col].iloc[0]:,.2f}"

        #Recommendation Box
        non_baseline = ranking_df[ranking_df['Scenarios'] != 'Baseline']
        if not non_baseline.empty:
            best_scenario = non_baseline.iloc[0]
            recommendation_text = f"{best_scenario['Scenarios']} is recommended based on highest ROI ({best_scenario['ROI']}) and fastest payback ({best_scenario['Payback Years']} years)."
        else:
            recommendation_text = "No investment scenarios added for comparison."

        base_tdc_display = f"R {base_tdc:,.0f}" if isinstance(base_tdc, (int, float)) else "N/A"
        #Get the context (corridor/disruption) from the first scenario
        report_corridor = results[0]['inputs_used']['corridor'] if results and 'inputs_used' in results[
            0] and 'corridor' in results[0]['inputs_used'] else corridor_options[0]
        report_disruption = results[0]['inputs_used']['disruption'] if results and 'inputs_used' in results[
            0] and 'disruption' in results[0]['inputs_used'] else disruption_options[0]
        table_cols = ['Scenarios', 'Resilience loss (Re)', 'TDC (R)', 'ROI', 'Payback Years', 'Details']

        #Final Report Layout Assembly
        card_style = {'backgroundColor': COLOR_CARD_BACKGROUND, 'borderColor': COLOR_BORDER_LIGHT}
        card_class = "shadow"
        return html.Div([
            dbc.Row([
                dbc.Col(html.H1("Scenario Analysis Report", style={'color': 'black'}), width="auto"),
                dbc.Col(dcc.Dropdown(options=[report_corridor], value=report_corridor, clearable=False, disabled=True,
                                     className="custom-input"), className="ms-auto", width=3),
                dbc.Col(
                    dcc.Dropdown(options=[report_disruption], value=report_disruption, clearable=False, disabled=True,
                                 className="custom-input"), width=3)
            ], align="center", className="mb-4"),
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardBody(
                    [html.P("Cost of unit Resilience loss (Cu)", style={'color': COLOR_METRIC_TEXT}),
                     html.H4(f"R {fin_in['Cu']:,.2f}", style={'color': COLOR_ACCENT_RED, 'fontWeight': 'bold'})])],
                    style=card_style, className=card_class)),  # Header removed
                dbc.Col(dbc.Card([dbc.CardBody(
                    [html.P("Avg. Resilience Loss (Re) - Baseline", style={'color': COLOR_METRIC_TEXT}),
                     html.H4(f"{base_re:,.2f}", style={'color': COLOR_ACCENT_RED, 'fontWeight': 'bold'})])],
                    style=card_style, className=card_class)),  # Header removed
                dbc.Col(dbc.Card([dbc.CardBody([html.P("Est. Annual Frequency", style={'color': COLOR_METRIC_TEXT}),
                                                html.H4(f"~{fin_in['frequency_per_year']}",
                                                        style={'color': COLOR_ACCENT_RED, 'fontWeight': 'bold'})])],
                                 style=card_style, className=card_class)),  # Header removed
                dbc.Col(dbc.Card([dbc.CardBody(
                    [html.P("Avg. Disruption Cost (TDC) - Baseline", style={'color': COLOR_METRIC_TEXT}),
                     html.H4(base_tdc_display, style={'color': COLOR_ACCENT_RED, 'fontWeight': 'bold'})])],
                    style=card_style, className=card_class))  # Header removed
            ], className="mb-4 text-center"),
            dbc.Row([
                dbc.Col(
                    dbc.Card([dbc.CardBody(dcc.Graph(figure=fig_curves, style={'height': '350px'}))], style=card_style,
                             className=card_class), width=7),
                dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=bcr_fig, style={'height': '350px'}))], style=card_style,
                                 className=card_class), width=5)
            ], className="mb-4"),
            html.H4("Scenario Ranking (Best to Worst)", className="mt-4", style={'color': 'black'}),
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardBody(
                    dbc.Table.from_dataframe(ranking_df[table_cols], striped=True, bordered=True, hover=True,
                                             className="m-0"))], style=card_style, className=card_class), width=8),
                dbc.Col(dbc.Card(dbc.CardBody([html.H5("Recommended Action", className="card-title"),
                                               html.P(recommendation_text,
                                                      style={'color': COLOR_PRIMARY_GREEN, 'fontWeight': 'bold'})]),
                                 style=card_style, className=card_class), width=4)
            ], className="mt-2")
        ])

    return create_report_page(stored_data, financial_inputs, baseline_re, baseline_tdc)


@app.callback(
    Output("page-content", "children"),
    Output("nav-links", "children"),
    Input("url", "pathname")
)
def render_page_content(pathname):
    """
    This is the main callback for page navigation.
    It reads the URL and returns the correct page layout.
    It also updates the "active" link in the sidebar.
    """
    active_style = {"backgroundColor": "white", "color": COLOR_PRIMARY_GREEN, "fontWeight": "bold",
                    "borderRadius": "0.3rem"}
    inactive_style = {"color": "white"}

    #Define all navigation links
    nav_links = [
        dbc.NavLink("Home", href="/", style=active_style if pathname == "/" else inactive_style),
        dbc.NavLink("Disruptions", href="/disruptions",
                    style=active_style if pathname == "/disruptions" else inactive_style),
        dbc.NavLink("Past Scenarios", href="/scenarios",
                    style=active_style if pathname == "/scenarios" else inactive_style),
        dbc.NavLink("Scenario Analysis", href="/inputs",
                    style=active_style if pathname == "/inputs" else inactive_style),
        dbc.NavLink("Report", href="/report", style=active_style if pathname == "/report" else inactive_style),
        dbc.NavLink("Detailed Analysis", href="/details",
                    style=active_style if pathname.startswith('/details') else inactive_style),
        dbc.NavLink("Settings", href="/settings", style=active_style if pathname == "/settings" else inactive_style)
    ]

    #Return the correct layout based on the URL
    if pathname == "/inputs":
        return inputs_page_layout, nav_links
    elif pathname == "/report":
        return report_page_layout, nav_links
    elif pathname and pathname.startswith('/details'):
        return details_page_layout, nav_links

    #Default to the inputs page
    return inputs_page_layout, nav_links


@app.callback(
    Output('details-content', 'children'),
    Input('url', 'pathname'),
    State('scenario-input-store', 'data')
)
def update_details_page(pathname, stored_inputs):
    """
    This callback generates the "Detailed Analysis" page for a single scenario.
    It's triggered when the URL changes to /details/....
    """
    if not pathname or not pathname.startswith('/details/'):
        return html.Div("Select a scenario's 'View Details' link from the Report page.")

    try:
        scenario_name = urllib.parse.unquote(pathname.split('/details/')[1])
    except IndexError:
        return html.Div("Invalid scenario name in URL.")

    scenario_inputs = stored_inputs.get(scenario_name)
    if not scenario_inputs:
        return html.Div(f"Could not find input data for scenario: {scenario_name}. Please add scenarios first.")

    corridor = scenario_inputs.get('corridor')
    disruption = scenario_inputs.get('disruption')
    if not corridor or not disruption:
        return html.Div(f"Missing context (corridor/disruption) for scenario: {scenario_name}")

    # Re-run the historical analysis to get the baseline archetype for this scenario.
    fitted_data, archetype, avg_ttr, avg_baseline_re_from_archetype, count, _, avg_adp_hours = run_historical_analysis(
        corridor, disruption)
    if archetype is None:
        return html.Div(f"Could not retrieve baseline data for {corridor}/{disruption}.")

    # Helper function to safely get numerical values from the input dictionary
    def get_num_value(data_dict, key, default=0):
        val = data_dict.get(key)
        if val is None: return default
        try:
            num_val = float(val)
            return num_val if isinstance(num_val, (int, float)) else default
        except (ValueError, TypeError):
            return default

    #3-Case Analysis (Optimistic, Most Likely, Pessimistic)
    results_list = []
    case_types = ["Optimistic", "Most Likely", "Pessimistic"]
    key_map = {"Optimistic": "min", "Most Likely": "mode", "Pessimistic": "max"}
    benefit_key_map = {"Optimistic": "max", "Most Likely": "mode", "Pessimistic": "min"}
    mode_params = {}

    for case in case_types:
        cost_key = key_map[case]
        benefit_key = benefit_key_map[case]

        #Get frequency for this case (min/mode/max)
        freq_val = get_num_value(scenario_inputs.get('disruption_frequency', {}), key_map[case],
                                 get_num_value(scenario_inputs.get('disruption_frequency', {}), 'mode'))
        annual_freq = None
        if freq_val is not None:
            try:
                annual_freq = float(freq_val) * 12  #Convert monthly to annual
            except (ValueError, TypeError):
                annual_freq = None

         #Build the parameter dictionary for this case
        params = {"name": f"{scenario_name} ({case})",
                  "duration_years": get_num_value(scenario_inputs, 'duration_years', 1),
                  "implementation_cost": get_num_value(scenario_inputs.get('implementation_cost', {}), cost_key,
                                                       get_num_value(scenario_inputs.get('implementation_cost', {}),
                                                                     'mode')),
                  "annual_opex": get_num_value(scenario_inputs.get('annual_opex', {}), cost_key,
                                               get_num_value(scenario_inputs.get('annual_opex', {}), 'mode')),
                  "perf_loss_reduction": get_num_value(scenario_inputs.get('perf_loss_reduction', {}), benefit_key,
                                                       get_num_value(scenario_inputs.get('perf_loss_reduction', {}),
                                                                     'mode')),
                  "frequency_per_year": annual_freq,  # Pass annual frequency
                  "new_adp_days": get_num_value(scenario_inputs.get('new_adp_days', {}), cost_key,
                                                get_num_value(scenario_inputs.get('new_adp_days', {}), 'mode')),
                  "new_ttr_days": get_num_value(scenario_inputs.get('new_ttr_days', {}), cost_key,
                                                get_num_value(scenario_inputs.get('new_ttr_days', {}), 'mode'))}

        if case == "Most Likely":
            mode_params = params.copy()  #Store the "mode" case for sensitivity analysis

        #Run the analysis for this case
        result = analyze_investment_scenario(archetype, avg_ttr, avg_adp_hours, financial_inputs, params,
                                             avg_baseline_re=avg_baseline_re_from_archetype)
        results_list.append(result)

    optimistic_result, likely_result, pessimistic_result = results_list
    baseline_roi = likely_result['business_case']['ROI']  #"Most Likely" ROI

    #Performance Range Plot
    fig_range = go.Figure()
    fig_range.add_trace(
        go.Scatter(x=likely_result['baseline_curve']['x'], y=likely_result['baseline_curve']['y'], mode='lines',
                   line=dict(color=COLOR_GRAPH_BASELINE_LINE, dash='dash', width=2), name='Baseline'))
    fig_range.add_trace(
        go.Scatter(x=optimistic_result['scenario_curve']['x'], y=optimistic_result['scenario_curve']['y'], mode='lines',
                   line=dict(color=COLOR_GRAPH_OPTIMISTIC, width=2, dash='dot'), name='Optimistic'))
    fig_range.add_trace(
        go.Scatter(x=likely_result['scenario_curve']['x'], y=likely_result['scenario_curve']['y'], mode='lines',
                   line=dict(color=COLOR_GRAPH_LIKELY, width=3), name='Most Likely'))
    fig_range.add_trace(
        go.Scatter(x=pessimistic_result['scenario_curve']['x'], y=pessimistic_result['scenario_curve']['y'],
                   mode='lines', line=dict(color=COLOR_GRAPH_PESSIMISTIC, width=2, dash='dot'), name='Pessimistic'))

    fig_range.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),  # Adjusted margins
        template='plotly_white', legend_title_text='Cases', yaxis_tickformat='.0%',
        plot_bgcolor=COLOR_CARD_BACKGROUND, paper_bgcolor=COLOR_CARD_BACKGROUND,
        xaxis_showgrid=False, yaxis_showgrid=False  # Remove gridlines
    )

    #Metric Ranges Table
    table_data = [optimistic_result['business_case'], likely_result['business_case'],
                  pessimistic_result['business_case']]
    range_df = pd.DataFrame(table_data)
    #Format the numbers for display
    for col in ['Resilience loss (Re)', 'TDC (R)', 'ROI', 'Payback Years']:
        if col in range_df.columns:
            if col == 'ROI':
                range_df[col] = range_df[col].apply(
                    lambda x: f"{x:,.1f}%" if isinstance(x, (int, float)) and np.isfinite(x) else (
                        "inf" if x == float('inf') else "N/A"))
            elif col == 'Payback Years':
                range_df[col] = range_df[col].apply(
                    lambda x: f"{x:,.1f}" if isinstance(x, (int, float)) and np.isfinite(x) else (
                        "inf" if x == float('inf') else "N/A"))
            else:
                range_df[col] = range_df[col].apply(
                    lambda x: f"R {x:,.0f}" if col == 'TDC (R)' and isinstance(x, (int, float)) else (
                        f"{x:,.2f}" if isinstance(x, (int, float)) else x))
    range_df = range_df[['Scenarios', 'Resilience loss (Re)', 'TDC (R)', 'ROI', 'Payback Years']]

    #Sensitivity (Tornado) Analysis
    sensitivity_data = []
    variables_to_test = [
        {"key": "implementation_cost", "label": "CAPEX", "type": "cost"},
        {"key": "annual_opex", "label": "OPEX", "type": "cost"},
        {"key": "perf_loss_reduction", "label": "Perf. Loss Reduction (%)", "type": "benefit"},
        {"key": "disruption_frequency", "label": "Monthly Frequency", "type": "cost"},
        {"key": "new_adp_days", "label": "Adaptation Time (Days)", "type": "cost"},
        {"key": "new_ttr_days", "label": "TTR (Days)", "type": "cost"}
    ]

    for var in variables_to_test:
        var_key = var["key"]
        var_label = var["label"]
        var_type = var["type"]

        mode_val_sens = get_num_value(scenario_inputs.get(var_key, {}), 'mode')
        if mode_val_sens is None:
            continue  #Skip if mode wasn't provided

        val_min_sens = get_num_value(scenario_inputs.get(var_key, {}), 'min', mode_val_sens)
        val_max_sens = get_num_value(scenario_inputs.get(var_key, {}), 'max', mode_val_sens)

        if val_min_sens == val_max_sens:
            continue  #Skip if no range was provided

        #Recalculate scenario with the "min" value for this variable
        min_params_sens = mode_params.copy()  #Start with "Most Likely" params
        min_params_sens[var_key] = val_min_sens
        if var_key == "disruption_frequency":  #Handle frequency conversion
            try:
                min_params_sens["frequency_per_year"] = float(val_min_sens) * 12
            except:
                min_params_sens["frequency_per_year"] = mode_params.get("frequency_per_year")  #Fallback

        min_result_sens = analyze_investment_scenario(archetype, avg_ttr, avg_adp_hours, financial_inputs,
                                                      min_params_sens, avg_baseline_re=avg_baseline_re_from_archetype)
        roi_at_min_input = min_result_sens['business_case']['ROI']

        #Recalculate scenario with the "max" value for this variable
        max_params_sens = mode_params.copy()
        max_params_sens[var_key] = val_max_sens
        if var_key == "disruption_frequency":  #Handle frequency conversion
            try:
                max_params_sens["frequency_per_year"] = float(val_max_sens) * 12
            except:
                max_params_sens["frequency_per_year"] = mode_params.get("frequency_per_year")  #Fallback

        max_result_sens = analyze_investment_scenario(archetype, avg_ttr, avg_adp_hours, financial_inputs,
                                                      max_params_sens, avg_baseline_re=avg_baseline_re_from_archetype)
        roi_at_max_input = max_result_sens['business_case']['ROI']

        #Determine which is "low" vs "high" based on whether it's a cost or benefit
        low_roi_val, high_roi_val = (roi_at_max_input, roi_at_min_input) if var_type == 'cost' else (roi_at_min_input,
                                                                                                     roi_at_max_input)

        if not np.isfinite(low_roi_val): low_roi_val = -1000
        if not np.isfinite(high_roi_val): high_roi_val = 1000

        sensitivity_data.append({'label': var_label, 'low_roi': low_roi_val, 'high_roi': high_roi_val,
                                 'impact': abs(high_roi_val - low_roi_val)})

    #Tornado Chart
    tornado_fig = go.Figure()
    if sensitivity_data:
        sensitivity_data.sort(key=lambda d: d['impact'], reverse=False)  #Sort by smallest impact
        fig_data = []
        for d in sensitivity_data:
            finite_baseline_roi = baseline_roi if np.isfinite(baseline_roi) else 0
            #Calculate impact relative to the baseline ROI
            low_impact = d['low_roi'] - finite_baseline_roi
            high_impact = d['high_roi'] - finite_baseline_roi
            fig_data.append({'Variable': d['label'], 'Impact': low_impact, 'Range': 'Low Estimate'})
            fig_data.append({'Variable': d['label'], 'Impact': high_impact, 'Range': 'High Estimate'})

        tornado_df = pd.DataFrame(fig_data)
        if not tornado_df.empty:
            tornado_fig = px.bar(tornado_df, y='Variable', x='Impact', color='Range', barmode='relative',
                                 orientation='h', title="Sensitivity Analysis on ROI",
                                 labels={'Impact': 'Change in Total ROI (%)', 'Variable': 'Input Parameter'},
                                 color_discrete_map={'Low Estimate': COLOR_GRAPH_PESSIMISTIC,
                                                     'High Estimate': COLOR_GRAPH_OPTIMISTIC})
            tornado_fig.update_layout(
                template='plotly_white', margin=dict(l=150),  #Add left margin for labels
                plot_bgcolor=COLOR_CARD_BACKGROUND, paper_bgcolor=COLOR_CARD_BACKGROUND,
                xaxis_showgrid=False, yaxis_showgrid=False  #Remove gridlines
            )
            tornado_fig.add_vline(x=0, line_dash="dash", line_color="black")
        else:
            tornado_fig = go.Figure().update_layout(title="Sensitivity Analysis (No input ranges provided)",
                                                    plot_bgcolor=COLOR_CARD_BACKGROUND,
                                                    paper_bgcolor=COLOR_CARD_BACKGROUND, xaxis_showgrid=False,
                                                    yaxis_showgrid=False)
    else:
        tornado_fig = go.Figure().update_layout(title="Sensitivity Analysis (No input ranges provided)",
                                                plot_bgcolor=COLOR_CARD_BACKGROUND, paper_bgcolor=COLOR_CARD_BACKGROUND,
                                                xaxis_showgrid=False, yaxis_showgrid=False)

    # Key Insights Card
    insights_content = [html.H5("Key Insights", className="card-title"),
                        html.P("This chart shows which input variable has the biggest impact on your final ROI.")]
    if sensitivity_data:
        sensitivity_data.sort(key=lambda d: d['impact'], reverse=True)  #Sort by largest impact
        most_sensitive = sensitivity_data[0]['label']
        total_impact = sensitivity_data[0]['impact']
        insights_content.append(html.P(
            [html.B("Finding: "), f"Your business case is ", html.Strong(f"most sensitive to {most_sensitive}."),
             f" The uncertainty in this one estimate alone can swing your ROI by approximately {total_impact:,.0f} percentage points."]))
        if len(sensitivity_data) > 1:
            second_sensitive = sensitivity_data[1]['label']
            insights_content.append(html.P([html.B("Recommendation: "),
                                            f"To build a reliable business case, you must focus on getting a more accurate estimate for {most_sensitive}. ",
                                            f"Changes in {second_sensitive} also have an impact, but are less critical."]))
        else:
            insights_content.append(html.P([html.B("Recommendation: "),
                                            f"To build a reliable business case, you must focus on getting a more accurate estimate for {most_sensitive}."]))
    else:
        insights_content.append(
            html.P("No sensitivity data was generated, likely because no min/max ranges were provided for any inputs."))

    insights_card = dbc.Card([dbc.CardBody(insights_content)],
                             style={'backgroundColor': COLOR_CARD_BACKGROUND, 'borderColor': COLOR_BORDER_LIGHT},
                             className="shadow")  #Add style and shadow

    #Final Detailed Page Layout Assembly
    card_style = {'backgroundColor': COLOR_CARD_BACKGROUND, 'borderColor': COLOR_BORDER_LIGHT}
    card_class = "shadow"
    return html.Div([
        html.H1(f"Detailed Analysis: {scenario_name}", style={'color': 'black'}),
        html.Hr(),
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=fig_range))], style=card_style, className=card_class),
                    width=7),
            dbc.Col(dbc.Card([dbc.CardHeader("Key Metric Ranges"), dbc.CardBody(
                dbc.Table.from_dataframe(range_df, striped=True, bordered=True, hover=True))], style=card_style,
                             className=card_class), width=5)
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=tornado_fig))], style=card_style, className=card_class),
                    width=8),
            dbc.Col(insights_card, width=4)
        ], className="mb-4"),
        html.Hr(),
        dbc.Button("Back to Report", href="/report", color="secondary")
    ])


# ==============================================================================
# 4. RUN THE APP
# ==============================================================================

if __name__ == '__main__':
    app.run(debug=True, port=8056)
