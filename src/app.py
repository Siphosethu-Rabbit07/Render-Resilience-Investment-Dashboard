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

# --- run_historical_analysis (MODIFIED Re calculation) ---
def run_historical_analysis(corridor_filter, disruption_filter):

    #Connects to DB, fetches data, isolates cycles, calculates TTR/ADP averages,
    #generates a smoothed archetype curve, and calculates baseline Re from that archetype.
    #Returns 7 values.

    # Initialize return values
    fitted_cycles_data = []  # Store individual fitted cycle data {x_hours, y_perf}
    final_archetype_curve = None
    avg_ttr_days = 0
    avg_baseline_re = 0  # Will be calculated from the final archetype
    cycle_count = 0
    min_perf = 0
    avg_adp_hours = 0

    try:
        db_params = {
            "user": "postgres", "password": "Sipho$e2", "host": "localhost",
            "port": "5432", "database": "tfr_resilience_db4"
        }
        engine = create_engine(
            f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}")

        # --- SQL Query ---
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

        # --- Performance Calculation ---
        df_4hr['performance'] = ((df_4hr['successful_trips'] / df_4hr['total_trips']).fillna(1.0)).clip(upper=1.0)
        df_4hr['time_block'] = pd.to_datetime(df_4hr['time_block'])

        # --- Cycle Isolation ---
        PERFORMANCE_THRESHOLD = 0.90
        BASELINE_RECOVERY_THRESHOLD = 0.93
        PADDING = 4
        in_disruption = False
        disruption_cycles_raw_padded = []
        actual_ttrs_days = []
        adaptation_durations_hours = []
        cycle_start_index = -1

        for i in range(len(df_4hr)):
            row = df_4hr.iloc[i] # Get the row using the integer position
            is_below_threshold = row['performance'] < PERFORMANCE_THRESHOLD
            if not in_disruption and is_below_threshold and row['is_selected_disruption_block'] == 1:
                in_disruption = True
                cycle_start_index = i
            elif in_disruption and not is_below_threshold:
                in_disruption = False
                cycle_end_index = i
                if cycle_start_index != -1:
                    unpadded_cycle_df = df_4hr.iloc[cycle_start_index:cycle_end_index]
                    if not unpadded_cycle_df.empty and unpadded_cycle_df['is_selected_disruption_block'].any():
                        recovery_end_index = cycle_end_index
                        while recovery_end_index < len(df_4hr) - 1 and df_4hr.iloc[recovery_end_index][
                            'performance'] < BASELINE_RECOVERY_THRESHOLD:
                            recovery_end_index += 1
                        end_slice = min(len(df_4hr), recovery_end_index + PADDING)
                        actual_start_time = df_4hr.iloc[cycle_start_index]['time_block']
                        actual_end_time = df_4hr.iloc[recovery_end_index]['time_block']
                        ttr_hours = (actual_end_time - actual_start_time).total_seconds() / 3600
                        if ttr_hours > 4: actual_ttrs_days.append(ttr_hours / 24)
                        start_slice = max(0, cycle_start_index - PADDING)
                        cycle_to_add = df_4hr.iloc[start_slice:end_slice].copy()
                        disruption_cycles_raw_padded.append(cycle_to_add)
                        # Calc ADP
                        y_perf_temp = unpadded_cycle_df['performance']
                        min_perf_value_temp = y_perf_temp.min()
                        trough_threshold_temp = min_perf_value_temp + 0.10
                        trough_points_temp = y_perf_temp[y_perf_temp <= trough_threshold_temp]
                        if not trough_points_temp.empty:
                            try:
                                true_trough_level_temp = trough_points_temp.mode()[0]
                                adaptation_points_indices_temp = y_perf_temp[
                                    y_perf_temp == true_trough_level_temp].index
                                if len(adaptation_points_indices_temp) > 0:
                                    adaptation_start_time_temp = df_4hr.loc[adaptation_points_indices_temp[0]][
                                        'time_block']
                                    adaptation_end_time_temp = df_4hr.loc[adaptation_points_indices_temp[-1]][
                                        'time_block']
                                    adp_duration_h = max((
                                                                     adaptation_end_time_temp - adaptation_start_time_temp).total_seconds() / 3600.0,
                                                         4.0)
                                    adaptation_durations_hours.append(adp_duration_h)
                            except IndexError:
                                pass
                cycle_start_index = -1

        # Handle disruption running to end
        if in_disruption and cycle_start_index != -1:
            unpadded_cycle_df = df_4hr.iloc[cycle_start_index:]
            if not unpadded_cycle_df.empty and unpadded_cycle_df['is_selected_disruption_block'].any():
                recovery_end_index = len(df_4hr) - 1
                end_slice = len(df_4hr)
                actual_start_time = df_4hr.iloc[cycle_start_index]['time_block']
                actual_end_time = df_4hr.iloc[recovery_end_index]['time_block']
                ttr_hours = (actual_end_time - actual_start_time).total_seconds() / 3600
                if ttr_hours > 4: actual_ttrs_days.append(ttr_hours / 24)
                start_slice = max(0, cycle_start_index - PADDING)
                disruption_cycles_raw_padded.append(df_4hr.iloc[start_slice:end_slice].copy())
                # Calc ADP
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
                        pass
        # --- End Cycle Isolation ---

        avg_ttr_days = np.mean(actual_ttrs_days) if actual_ttrs_days else 0
        avg_adp_hours = np.mean(adaptation_durations_hours) if adaptation_durations_hours else 0
        cycle_count = len(disruption_cycles_raw_padded)
        print(
            f"[Core Logic] Identified {cycle_count} cycles. Avg TTR: {avg_ttr_days:.2f} days. Avg Adp: {avg_adp_hours:.2f} hrs.")

        if cycle_count == 0:
            print(f"No disruption cycles isolated.")
            return [], None, 0, 0, 0, 0, 0

        # --- Fitting Logic ---
        # (Fitting logic remains the same - applies polynomials to each padded cycle)
        # ... (Omitted for brevity, but assumes fitted_cycles_data is populated like before) ...
        fitted_cycles_data = []  # Reset/ensure empty before loop
        for cycle_df in disruption_cycles_raw_padded:
            if len(cycle_df) < 5: continue
            cycle_df['time_block'] = pd.to_datetime(cycle_df['time_block'])
            x_time_hours_relative = (cycle_df['time_block'] - cycle_df['time_block'].iloc[
                0]).dt.total_seconds() / 3600.0
            y_perf = cycle_df['performance']
            min_perf_value = y_perf.min()
            if min_perf_value > 0.98: continue
            trough_threshold = min_perf_value + 0.10
            trough_points = y_perf[y_perf <= trough_threshold]
            if trough_points.empty: continue
            try:
                true_trough_level = trough_points.mode()[0]
                adaptation_points_indices = y_perf[y_perf == true_trough_level].index
                if len(adaptation_points_indices) == 0: continue
                adaptation_start_idx = adaptation_points_indices[0]
                adaptation_end_idx = adaptation_points_indices[-1]
                relative_start_loc = cycle_df.index.get_loc(adaptation_start_idx)
                relative_end_loc = cycle_df.index.get_loc(adaptation_end_idx)
                if relative_start_loc >= relative_end_loc:
                    if relative_start_loc == relative_end_loc:
                        relative_end_loc = relative_start_loc + 1
                    else:
                        continue
            except (IndexError, KeyError) as e:
                continue
            absorption_df = cycle_df.iloc[:relative_start_loc + 1]
            adaptation_df = cycle_df.iloc[relative_start_loc:min(relative_end_loc + 1, len(cycle_df))]
            recovery_df = cycle_df.iloc[min(relative_end_loc + 1, len(cycle_df)):]
            if len(absorption_df) < 2 or len(adaptation_df) < 1 or len(recovery_df) < 2: continue
            try:
                X_abs = x_time_hours_relative.iloc[:relative_start_loc + 1].values.reshape(-1, 1)
                X_adp = x_time_hours_relative.iloc[
                    relative_start_loc:min(relative_end_loc + 1, len(cycle_df))].values.reshape(-1, 1)
                X_rec = x_time_hours_relative.iloc[min(relative_end_loc + 1, len(cycle_df)):].values.reshape(-1, 1)
                if X_abs.size == 0 or X_adp.size == 0 or X_rec.size == 0: continue
                abs_poly = PolynomialFeatures(degree=2)
                X_abs_poly = abs_poly.fit_transform(X_abs)
                abs_model = LinearRegression().fit(X_abs_poly, absorption_df['performance'].values)
                adp_poly = PolynomialFeatures(degree=1 if len(adaptation_df) > 1 else 0)
                X_adp_poly = adp_poly.fit_transform(X_adp)
                adp_model = LinearRegression().fit(X_adp_poly, adaptation_df['performance'].values)
                rec_poly = PolynomialFeatures(degree=2)
                X_rec_poly = rec_poly.fit_transform(X_rec)
                rec_model = LinearRegression().fit(X_rec_poly, recovery_df['performance'].values)
                x_smooth_abs = np.linspace(X_abs.min(), X_abs.max(), 50).reshape(-1, 1)
                y_smooth_abs = abs_model.predict(abs_poly.transform(x_smooth_abs))
                x_smooth_adp_rel = np.linspace(X_adp.min(), X_adp.max(), 50).reshape(-1, 1)
                y_smooth_adp = adp_model.predict(adp_poly.transform(x_smooth_adp_rel))
                x_smooth_rec_rel = np.linspace(X_rec.min(), X_rec.max(), 100).reshape(-1, 1)
                y_smooth_rec = rec_model.predict(rec_poly.transform(x_smooth_rec_rel))
                if len(y_smooth_abs) > 0 and len(y_smooth_adp) > 0 and x_smooth_abs[-1, 0] >= x_smooth_adp_rel[0, 0]:
                    y_smooth_abs[-1] = y_smooth_adp[0]
                if len(y_smooth_adp) > 0 and len(y_smooth_rec) > 0 and x_smooth_adp_rel[-1, 0] >= x_smooth_rec_rel[
                    0, 0]: y_smooth_adp[-1] = y_smooth_rec[0]
                combined_x = np.concatenate(
                    [x_smooth_abs.flatten(), x_smooth_adp_rel.flatten(), x_smooth_rec_rel.flatten()])
                combined_y = np.concatenate([y_smooth_abs.flatten(), y_smooth_adp.flatten(), y_smooth_rec.flatten()])
                unique_indices = np.unique(combined_x, return_index=True)[1]
                combined_x = combined_x[unique_indices]
                combined_y = combined_y[unique_indices]
                sort_order = np.argsort(combined_x)
                combined_x = combined_x[sort_order]
                combined_y = combined_y[sort_order]
                fitted_cycles_data.append({"x": combined_x, "y": combined_y})  # Changed variable name
            except ValueError as fit_error:
                continue

        if not fitted_cycles_data:  # Changed variable name
            print(f"No cycles could be successfully fitted.")
            return [], None, avg_ttr_days, 0, cycle_count, 0, avg_adp_hours

        # --- Archetype Generation ---
        normalized_x = np.linspace(0, 100, 101)
        all_normalized_curves = []
        for cycle in fitted_cycles_data:  # Changed variable name
            original_x, original_y = cycle['x'], cycle['y']
            duration = original_x.max() - original_x.min()
            if duration > 0: all_normalized_curves.append(
                interp1d((original_x - original_x.min()) / duration * 100, original_y, bounds_error=False,
                         fill_value="extrapolate")(normalized_x))
        if not all_normalized_curves: return fitted_cycles_data, None, avg_ttr_days, 0, cycle_count, 0, avg_adp_hours  # Changed variable name
        archetypal_curve_y = np.array(all_normalized_curves).mean(axis=0)

        # --- Final Smoothing ---
        final_poly = PolynomialFeatures(degree=7)
        X_final_poly = final_poly.fit_transform(normalized_x.reshape(-1, 1))
        final_model = LinearRegression().fit(X_final_poly, archetypal_curve_y)
        final_smooth_y = final_model.predict(X_final_poly)
        final_smooth_y = np.clip(final_smooth_y, 0, 1.0)

        final_archetype_curve = {"x": normalized_x, "y": final_smooth_y}

        # --- Calculate Baseline Re from the FINAL ARCHETYPE scaled to AVG TTR ---
        base_ttr_hours_for_re = avg_ttr_days * 24 if avg_ttr_days > 0 else 24  # Use calculated avg TTR
        baseline_x_hours_for_re = final_archetype_curve['x'] * (base_ttr_hours_for_re / 100)
        archetype_y_for_re = np.clip(final_archetype_curve['y'], 0, 1.0)
        avg_baseline_re = np.trapz(1.0 - archetype_y_for_re, baseline_x_hours_for_re)
        # --- End Baseline Re Calculation ---

        min_perf = final_archetype_curve['y'].min() if final_archetype_curve else 0

        # Return 7 values (fitted_cycles_data is used internally but not directly by analyze_investment_scenario anymore for Re baseline)
        return fitted_cycles_data, final_archetype_curve, avg_ttr_days, avg_baseline_re, cycle_count, min_perf, avg_adp_hours

    except Exception as e:
        print(f"ERROR in run_historical_analysis: {e}")
        traceback.print_exc()
        return [], None, 0, 0, 0, 0, 0


# --- analyze_investment_scenario (CORRECTED to use archetype Re consistently) ---
def analyze_investment_scenario(fitted_cycles, final_archetype_curve, avg_ttr_days, avg_adp_hours, financial_params,
                                investment_params, avg_baseline_re):  # Added avg_baseline_re
    # fitted_cycles is no longer directly used for baseline Re calculation here
    if final_archetype_curve is None:  # Simplified check
        return {"baseline_curve": {"x": [], "y": [], "Re": 0, "TDC": 0}, "scenario_curve": {"x": [], "y": [], "Re": 0},
                "business_case": {"Scenarios": investment_params['name'], "Resilience loss (Re)": 0,
                                  "TDC (R)": investment_params.get('implementation_cost', 0), "ROI": -100.0,
                                  "Payback Years": float('inf'), "BCR": 0, "Benefit": 0,
                                  "Cost": investment_params.get('implementation_cost', 0)},
                "inputs_used": investment_params}

    BASELINE_PERFORMANCE = 1.0

    # --- Use avg_baseline_re passed in (now calculated from archetype in run_historical_analysis) ---
    archetype_baseline_re = avg_baseline_re  # Use the passed-in value
    archetype_baseline_tdc = financial_params['Cu'] * archetype_baseline_re  # TDC per event based on archetype baseline

    # --- Input Parameter Processing ---
    perf_loss_reduction = investment_params.get('perf_loss_reduction', 0) / 100.0 if investment_params.get(
        'perf_loss_reduction') is not None else 0.0
    scenario_ci = investment_params.get('implementation_cost', 0) if investment_params.get(
        'implementation_cost') is not None else 0.0
    annual_opex = investment_params.get('annual_opex', 0) if investment_params.get('annual_opex') is not None else 0.0
    duration_years = investment_params.get('duration_years', 1) if investment_params.get(
        'duration_years') is not None else 1.0
    frequency_per_year = investment_params.get('frequency_per_year')  # Get potentially updated frequency
    # Fallback if frequency wasn't passed or was invalid
    if frequency_per_year is None:
        frequency_per_year = financial_params.get('frequency_per_year', 0)

    # --- Time Parameter Adjustment ---
    new_ttr_days_input = investment_params.get('new_ttr_days', None)
    base_ttr_hours = avg_ttr_days * 24 if avg_ttr_days > 0 else 24

    new_adp_days_input = investment_params.get('new_adp_days', None)
    effective_avg_adp_hours = avg_adp_hours if avg_adp_hours > 0 else 4

    # --- Use baseline if input is None OR <= 0 ---
    user_ttr_hours = base_ttr_hours if new_ttr_days_input is None or new_ttr_days_input <= 0 else new_ttr_days_input * 24
    user_adp_hours = effective_avg_adp_hours if new_adp_days_input is None or new_adp_days_input <= 0 else new_adp_days_input * 24

    adp_duration_delta_hours = user_adp_hours - effective_avg_adp_hours
    final_new_ttr_hours = max(4, user_ttr_hours + adp_duration_delta_hours)

    # --- Scenario Curve Modification ---
    archetype_x_norm = final_archetype_curve['x']
    archetype_y = np.clip(final_archetype_curve['y'], 0, 1.0)
    modified_y = BASELINE_PERFORMANCE - ((BASELINE_PERFORMANCE - archetype_y) * (1 - perf_loss_reduction))
    modified_y = np.clip(modified_y, 0, 1.0)
    modified_x_hours = archetype_x_norm * (final_new_ttr_hours / 100)  # Scale x-axis to *scenario* TTR

    # --- Scenario Resilience Loss (Re) Calculation (using archetype and scenario TTR) ---
    scenario_re = np.trapz(BASELINE_PERFORMANCE - modified_y, modified_x_hours)
    scenario_tdc_event = financial_params['Cu'] * scenario_re # Scenario TDC per event

    # --- Financial Calculations (using ARCHETYPE BASELINE TDC for comparison) ---
    savings_per_event = archetype_baseline_tdc - scenario_tdc_event  # <-- CONSISTENT COMPARISON
    annual_savings = (savings_per_event * frequency_per_year) - annual_opex
    calc_duration = max(1, duration_years or 1)
    total_net_profit = (annual_savings * calc_duration) - scenario_ci
    total_roi = (total_net_profit / scenario_ci * 100) if scenario_ci > 0 else float('inf')
    payback_years = (scenario_ci / annual_savings) if annual_savings > 0 else float('inf')
    benefit_value = savings_per_event * frequency_per_year  # Annual gross benefit
    bcr = benefit_value / scenario_ci if scenario_ci > 0 else float('inf')  # Annual gross benefit / CAPEX

    # --- Return Dictionary ---
    # Baseline curve for visualization scaled to baseline TTR
    baseline_x_hours_visual = archetype_x_norm * (base_ttr_hours / 100)
    return {
        # Pass the archetype baseline Re and TDC for consistency in return structure if needed elsewhere
        "baseline_curve": {"x": baseline_x_hours_visual, "y": archetype_y, "Re": archetype_baseline_re,
                           "TDC": archetype_baseline_tdc},
        "scenario_curve": {"x": modified_x_hours, "y": modified_y, "Re": scenario_re},
        "business_case": {
            "Scenarios": investment_params['name'],
            "Resilience loss (Re)": scenario_re,
            "TDC (R)": scenario_tdc_event,  # Scenario TDC per event
            "ROI": total_roi,
            "Payback Years": payback_years,
            "BCR": bcr,
            "Benefit": benefit_value,  # Annual Gross Benefit
            "Cost": scenario_ci},
        "inputs_used": investment_params}


# This function is not called by the app, but is available
# (create_tornado_chart - Assumed unchanged, omitted for brevity)
# ...

# ==============================================================================
# 2. APP INITIALIZATION & LAYOUT DEFINITIONS
# ==============================================================================
financial_inputs = {"Cu": 90138.81, "frequency_per_year": 12}
# (Color definitions unchanged - ADDED COLOR_GRAPH_BASELINE_LINE)
# ...
COLOR_PRIMARY_GREEN: str = '#2E8B57'
COLOR_LIGHT_GREEN_F = '#90EE90'  # Light green for 'F' in title
COLOR_ACCENT_RED = '#DC3545'
COLOR_CARD_BACKGROUND = '#D9F2D0'
COLOR_INPUT_BACKGROUND = 'rgba(46, 139, 87, 0.15)'  # Darker transparent green for inputs
COLOR_METRIC_TEXT = '#0D3512'  # Dark green for metric labels
COLOR_BORDER_LIGHT = '#E9ECEF'  # Light grey border for cards
COLOR_CHART_FILL_GREEN = 'rgba(46, 139, 87, 0.3)'  # Semi-transparent green for chart fill
COLOR_CHART_LINE_DARKGREEN = '#0D3512'  # Dark green for chart line
COLOR_GRAPH_BASELINE_LINE = COLOR_CHART_LINE_DARKGREEN  # Set baseline line color to dark green
# --- ADDED Missing Colors for Range Plot ---
COLOR_GRAPH_OPTIMISTIC = 'green'
COLOR_GRAPH_LIKELY = 'orange'
COLOR_GRAPH_PESSIMISTIC = 'red'
# --- End Added Colors ---
COLOR_PAGE_BACKGROUND = '#F2F2F2'  # Light grey page background
COLOR_TABLE_STRIPE = '#E9ECEF'  # Light grey for table stripe

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE], suppress_callback_exceptions=True)
server = app.server

# --- Custom CSS for Input Placeholders & Table Styling ---
# (custom_css definition unchanged)
# ...
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

# --- Define app.index_string to inject custom CSS ---
# (app.index_string definition unchanged)
# ...
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

# --- Dropdown Options ---
# (Unchanged)
# ...
corridor_options = ['North Corridor']
disruption_options = ['Cable Theft', 'Track Failure']

# --- Layouts ---
# (sidebar definition unchanged)
# ...
sidebar = html.Div(
    [
        # --- Updated Sidebar Title ---
        html.H3(  # Changed from H2 to H3 for smaller size
            ["Transnet ",
             html.Span("F", style={'color': COLOR_LIGHT_GREEN_F, 'fontWeight': 'bold'}),  # Lighter Green F
             html.Span("R", style={'color': COLOR_ACCENT_RED, 'fontWeight': 'bold'})],  # Red R
            className="text-center",  # Center align title
            style={'color': 'white', 'marginBottom': '0'}  # Added margin bottom 0
        ),
        html.Hr(style={'borderColor': 'white', 'marginTop': '0.5rem'}),  # Reduced margin top
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


# (make_input_group definition unchanged)
# ...
def make_input_group(label, mode_id):
    # Apply custom-input class and background style to dbc.Input
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


# (inputs_page_layout definition unchanged)
# ...
inputs_page_layout = html.Div([
    dbc.Row([
        dbc.Col(html.H1("Resilience Scenario Inputs", style={'color': 'black'}), width="auto"),  # Title back to black
        dbc.Col(dcc.Dropdown(id='corridor-filter', options=corridor_options, value=corridor_options[0], clearable=False,
                             className="custom-input"), className="ms-auto", width=3),  # Added custom-input class
        dbc.Col(dcc.Dropdown(id='disruption-filter', options=disruption_options, value=disruption_options[0],
                             clearable=False, className="custom-input"), width=3)  # Added custom-input class
    ], align="center", className="mb-4"),
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader(id='baseline-curve-title'),
            dbc.CardBody(dcc.Graph(id='baseline-curve-graph'))
        ], style={'backgroundColor': COLOR_CARD_BACKGROUND, 'borderColor': COLOR_BORDER_LIGHT}, className="shadow"),
            width=7),  # Added shadow and border
        dbc.Col([
            html.H4("Resilience Metrics", className="text-center mb-3"),
            # --- Reverted Metric Card Layout to Vertical Stack ---
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
                     className="shadow mb-3"),  # Added shadow, border, margin
            dbc.Card(id='ploss-card', children=dbc.CardBody([
                html.P("Avg. Performance loss (α)", style={'color': COLOR_METRIC_TEXT}),
                html.H4("0%", style={'color': COLOR_ACCENT_RED, 'fontWeight': 'bold'}, id='ploss-value-h4')
            ]), style={'backgroundColor': COLOR_CARD_BACKGROUND, 'borderColor': COLOR_BORDER_LIGHT}, className="shadow")
            # Added shadow, border
        ], width=5)
    ], className="mb-4"),
    # --- Applying text highlighting using html.Span ---
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

# (report_page_layout, details_page_layout definitions unchanged)
# ...
report_page_layout = html.Div(id="report-content")
details_page_layout = html.Div(id="details-content")

# (content definition unchanged)
# ...
content = html.Div(id="page-content",
                   style={'marginLeft': '15rem', 'padding': '2rem 1rem', 'backgroundColor': COLOR_PAGE_BACKGROUND})

# --- app.layout (Corrected - no html.Style here) ---
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Store(id='scenario-data-store', data=[]),
    dcc.Store(id='scenario-input-store', data={}),
    sidebar,
    content
])


# ==============================================================================
# 3. CALLBACKS
# ==============================================================================

# --- update_profile_callback (Unchanged from previous version) ---
# ...
@app.callback(
    Output('baseline-curve-graph', 'figure'),
    Output('baseline-curve-title', 'children'),
    Output('tdc-value-h4', 'children'),
    Output('ploss-value-h4', 'children'),
    Input('corridor-filter', 'value'),
    Input('disruption-filter', 'value')
)
def update_profile_callback(corridor, disruption):
    fitted, archetype, avg_ttr, avg_re, count, min_perf, avg_adp_hours = run_historical_analysis(corridor, disruption)

    tdc_value = "N/A"
    ploss_value = "N/A"
    # Default fig with background and no grid
    fig = go.Figure().update_layout(
        title_text="No Data Found or No Cycles Isolated", template='plotly_white',
        plot_bgcolor=COLOR_CARD_BACKGROUND, paper_bgcolor=COLOR_CARD_BACKGROUND,
        xaxis_showgrid=False, yaxis_showgrid=False  # Remove gridlines
    )
    title = "N/A"

    if archetype:
        x_hours_axis = archetype['x'] * (avg_ttr * 24 / 100) if avg_ttr > 0 else archetype['x']
        y_perf = archetype['y']
        y_baseline = np.ones_like(y_perf)

        fig = go.Figure()
        # Add the 100% baseline trace first (invisible, for fill)
        fig.add_trace(go.Scatter(x=x_hours_axis, y=y_baseline, mode='lines', line=dict(width=0), showlegend=False))
        # Add the performance curve, filling *up* to the baseline
        fig.add_trace(go.Scatter(
            x=x_hours_axis, y=y_perf, fill='tonexty', mode='lines',
            line=dict(color=COLOR_CHART_LINE_DARKGREEN, width=3),
            fillcolor=COLOR_CHART_FILL_GREEN, name='Performance'
        ))

        min_y_range = max(0, min_perf - 0.1 if min_perf is not None else 0.5)
        upper_y_range = 1.05

        fig.update_layout(
            # title_text=f"Re = {avg_re:,.2f}", # Remove title text for annotation
            xaxis_title="Time (Hours)",
            yaxis_range=[min_y_range, upper_y_range], yaxis_tickformat='.0%',
            margin=dict(t=30, l=40, r=20, b=40),  # Adjusted top margin
            template='plotly_white',
            plot_bgcolor=COLOR_CARD_BACKGROUND, paper_bgcolor=COLOR_CARD_BACKGROUND,
            showlegend=False,
            xaxis_showgrid=False, yaxis_showgrid=False  # Remove gridlines
        )

        # --- Add Annotation ---
        center_x_index = len(x_hours_axis) // 2
        annotation_x = x_hours_axis[center_x_index]
        # Position annotation slightly above the performance curve minimum in the center
        annotation_y = (y_perf[center_x_index] + 1.0) / 2  # Adjust vertical positioning

        fig.add_annotation(
            x=annotation_x, y=annotation_y,
            # Using HTML within text for multi-line and styling
            text=f"<span style='color:{COLOR_ACCENT_RED}; font-weight:bold; font-size: 14px;'>Re = {avg_re:,.2f}</span><br><span style='color:{COLOR_METRIC_TEXT}; font-size: 12px;'>Resilience Loss Index</span>",
            showarrow=False,
            align="center"
        )
        # --- End Annotation ---

        title = f"{corridor} Performance Under {disruption} Disruption ({count} cycles found)"
        # Use the avg_baseline_re (calculated from archetype) for TDC display consistency
        tdc_val_num = avg_re * financial_inputs['Cu'] if avg_re > 0 else 0
        tdc_value = f"R {tdc_val_num:,.0f}"
        perf_loss_num = (1.0 - min_perf) if min_perf is not None else 0
        perf_loss_num = max(0, perf_loss_num)
        ploss_value = f"{perf_loss_num:.0%}"

    return fig, title, tdc_value, ploss_value


# --- add_scenario_callback (MODIFIED to pass frequency) ---
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
     # Get frequency input
     State('at-min-input-min', 'value'), State('at-min-input', 'value'), State('at-min-input-max', 'value'),
     State('ttr-input-min', 'value'), State('ttr-input', 'value'), State('ttr-input-max', 'value'),
     State('corridor-filter', 'value'), State('disruption-filter', 'value')],
    prevent_initial_call=True
)
def add_scenario_callback(n_clicks, existing_results, existing_inputs, name, duration_years,
                          capex_min, capex_mode, capex_max, opex_min, opex_mode, opex_max,
                          ploss_min, ploss_mode, ploss_max,
                          reach_min_min, reach_min_mode, reach_min_max,
                          freq_min, freq_mode, freq_max,  # Get frequency input
                          adp_min, adp_mode, adp_max,
                          ttr_min, ttr_mode, ttr_max,
                          corridor, disruption):
    if not n_clicks: return no_update, no_update, no_update, no_update
    scenario_name = name or f"Scenario {len(existing_results) + 1}"
    # --- run_historical_analysis now returns avg_baseline_re calculated from archetype ---
    fitted_data, archetype, avg_ttr, avg_baseline_re_from_archetype, count, _, avg_adp_hours = run_historical_analysis(
        corridor, disruption)
    if archetype is None: return no_update, no_update, "Could not run analysis: No baseline data.", no_update

    # Store all inputs
    new_input_data = {scenario_name: {
        "name": scenario_name, "corridor": corridor, "disruption": disruption, "duration_years": duration_years,
        "implementation_cost": {"min": capex_min, "mode": capex_mode, "max": capex_max},
        "annual_opex": {"min": opex_min, "mode": opex_mode, "max": opex_max},
        "perf_loss_reduction": {"min": ploss_min, "mode": ploss_mode, "max": ploss_max},
        "time_to_reach_min": {"min": reach_min_min, "mode": reach_min_mode, "max": reach_min_max},
        "disruption_frequency": {"min": freq_min, "mode": freq_mode, "max": freq_max},  # Store frequency
        "new_adp_days": {"min": adp_min, "mode": adp_mode, "max": adp_max},
        "new_ttr_days": {"min": ttr_min, "mode": ttr_mode, "max": ttr_max}}}
    updated_inputs = {**existing_inputs, **new_input_data}

    # Pass Mode values to analyze_investment_scenario, including frequency
    mode_investment_params = {
        "name": scenario_name, "duration_years": duration_years, "implementation_cost": capex_mode or 0,
        "annual_opex": opex_mode or 0, "perf_loss_reduction": ploss_mode or 0,
        # "time_to_reach_min": reach_min_mode, # Pass if needed by analysis function
        "frequency_per_year": freq_mode,  # Pass frequency (might be monthly, convert if needed)
        "new_adp_days": adp_mode,
        "new_ttr_days": ttr_mode,
        "corridor": corridor, "disruption": disruption
    }
    # Convert monthly frequency to annual
    if mode_investment_params["frequency_per_year"] is not None:
        try:
            mode_investment_params["frequency_per_year"] = float(mode_investment_params["frequency_per_year"]) * 12
        except (ValueError, TypeError):
            print("Warning: Could not convert monthly frequency to annual. Using default.")
            mode_investment_params["frequency_per_year"] = None  # Let analyze_investment_scenario use default

    # --- Call analyze_investment_scenario with avg_baseline_re from archetype ---
    # Note: fitted_data (individual cycles) is not needed by analyze_investment_scenario anymore for Re baseline
    mode_result = analyze_investment_scenario(None, archetype, avg_ttr, avg_adp_hours, financial_inputs,
                                              mode_investment_params, avg_baseline_re=avg_baseline_re_from_archetype)
    updated_results = existing_results + [mode_result]
    scenario_badges = [dbc.Badge(s['business_case']['Scenarios'], color="primary", className="me-1 mb-1") for s in
                       updated_results]
    return updated_results, updated_inputs, scenario_badges, ''


# --- navigate_to_report (Unchanged logic) ---
# ...
@app.callback(Output('url', 'pathname', allow_duplicate=True), Input('run-btn', 'n_clicks'), prevent_initial_call=True)
def navigate_to_report(n_clicks):
    if n_clicks: return '/report'
    return no_update


# --- clear_scenarios_callback (Unchanged logic) ---
# ...
@app.callback(
    Output('scenario-data-store', 'data', allow_duplicate=True),
    Output('scenario-input-store', 'data', allow_duplicate=True),
    Output('scenarios-list', 'children', allow_duplicate=True),
    Input('clear-btn', 'n_clicks'), prevent_initial_call=True
)
def clear_scenarios_callback(n_clicks):
    return [], {}, []


# --- update_report_page_callback (Unchanged logic, styles already applied) ---
# ...
@app.callback(Output('report-content', 'children'), Input('scenario-data-store', 'data'))
def update_report_page_callback(stored_data):
    # (Input validation and data fetching logic unchanged)
    if not stored_data: return html.Div("Please add one or more scenarios from the 'Scenario Analysis' page first.")
    try:
        baseline_re = stored_data[0]['baseline_curve']['Re']; baseline_tdc = stored_data[0]['baseline_curve']['TDC']
    except (IndexError, KeyError, TypeError):
        return html.Div("Error accessing baseline data from stored scenarios.")

    # (create_report_page inner function logic mostly unchanged, applies styles)
    def create_report_page(results, fin_in, base_re, base_tdc):
        # --- Chart generation with background colors & no gridlines ---
        fig_curves = go.Figure()
        fig_curves.add_trace(
            go.Scatter(x=results[0]['baseline_curve']['x'], y=results[0]['baseline_curve']['y'], mode='lines',
                       line=dict(color=COLOR_GRAPH_BASELINE_LINE, dash='dash', width=2), name='Baseline'))
        for res in results: fig_curves.add_trace(
            go.Scatter(x=res['scenario_curve']['x'], y=res['scenario_curve']['y'], mode='lines', line=dict(width=3),
                       name=res['business_case']['Scenarios']))
        fig_curves.update_layout(
            margin=dict(l=20, r=20, t=40, b=20), template='plotly_white',  # Added top margin back for title
            legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
            yaxis_tickformat='.0%', title="Effects of Potential Resilience Investments on Performance",
            plot_bgcolor=COLOR_CARD_BACKGROUND, paper_bgcolor=COLOR_CARD_BACKGROUND,
            xaxis_showgrid=False, yaxis_showgrid=False  # Remove gridlines
        )

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
                          secondary_y=True)
        bcr_fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=40), barmode='group', template='plotly_white',
            # Increased bottom margin for legend
            title="Benefit vs Cost & BCR for Resilience Scenarios",
            plot_bgcolor=COLOR_CARD_BACKGROUND, paper_bgcolor=COLOR_CARD_BACKGROUND,
            xaxis_showgrid=False, yaxis_showgrid=False,  # Remove gridlines
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
            # Horizontal legend at bottom-center
        )
        bcr_fig.update_yaxes(title_text="Rands", secondary_y=False, showgrid=False)  # Remove gridlines
        bcr_fig.update_yaxes(title_text="BCR Value", secondary_y=True, showgrid=False)  # Remove gridlines

        # (Table generation logic unchanged)
        baseline_row = pd.DataFrame(
            [{"Scenarios": "Baseline", "Resilience loss (Re)": base_re, "TDC (R)": base_tdc, "ROI": 0.0,
              "Payback Years": float('inf'), "Details": ""}])
        try:
            results_df = pd.concat([pd.DataFrame([s['business_case']]) for s in results], ignore_index=True)
        except Exception as df_error:
            results_df = pd.DataFrame()
        if not results_df.empty:
            results_df['Details'] = results_df['Scenarios'].apply(
                lambda name: dcc.Link("View Details", href=f"/details/{urllib.parse.quote(name)}"))
            ranking_df = pd.concat([results_df, baseline_row], ignore_index=True).sort_values(by='ROI', ascending=False)
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
        non_baseline = ranking_df[ranking_df['Scenarios'] != 'Baseline']
        if not non_baseline.empty:
            best_scenario = non_baseline.iloc[
                0]; recommendation_text = f"{best_scenario['Scenarios']} is recommended based on highest ROI ({best_scenario['ROI']}) and fastest payback ({best_scenario['Payback Years']} years)."
        else:
            recommendation_text = "No investment scenarios added for comparison."
        base_tdc_display = f"R {base_tdc:,.0f}" if isinstance(base_tdc, (int, float)) else "N/A"
        report_corridor = results[0]['inputs_used']['corridor'] if results and 'inputs_used' in results[
            0] and 'corridor' in results[0]['inputs_used'] else corridor_options[0]
        report_disruption = results[0]['inputs_used']['disruption'] if results and 'inputs_used' in results[
            0] and 'disruption' in results[0]['inputs_used'] else disruption_options[0]
        table_cols = ['Scenarios', 'Resilience loss (Re)', 'TDC (R)', 'ROI', 'Payback Years', 'Details']

        # --- Apply styling to cards in the report layout (removed headers from metric cards) ---
        card_style = {'backgroundColor': COLOR_CARD_BACKGROUND, 'borderColor': COLOR_BORDER_LIGHT}
        card_class = "shadow"
        return html.Div([
            dbc.Row([dbc.Col(html.H1("Scenario Analysis Report", style={'color': 'black'}), width="auto"), dbc.Col(
                dcc.Dropdown(options=[report_corridor], value=report_corridor, clearable=False, disabled=True,
                             className="custom-input"), className="ms-auto", width=3), dbc.Col(
                dcc.Dropdown(options=[report_disruption], value=report_disruption, clearable=False, disabled=True,
                             className="custom-input"), width=3)], align="center", className="mb-4"),
            # Added custom-input class
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


# --- render_page_content (Unchanged logic, styling updated) ---
# ... (Omitted for brevity - unchanged from previous version) ...
@app.callback(
    Output("page-content", "children"),
    Output("nav-links", "children"),
    Input("url", "pathname")
)
def render_page_content(pathname):
    active_style = {"backgroundColor": "white", "color": COLOR_PRIMARY_GREEN, "fontWeight": "bold",
                    "borderRadius": "0.3rem"}
    inactive_style = {"color": "white"}
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
    if pathname == "/inputs":
        return inputs_page_layout, nav_links
    elif pathname == "/report":
        return report_page_layout, nav_links
    elif pathname and pathname.startswith('/details'):
        return details_page_layout, nav_links
    return inputs_page_layout, nav_links


# --- update_details_page (Applies card styling, shadows, removes gridlines) ---
# ... (Omitted for brevity - uses consistent archetype Re logic now) ...
@app.callback(
    Output('details-content', 'children'),
    Input('url', 'pathname'),
    State('scenario-input-store', 'data')
)
def update_details_page(pathname, stored_inputs):
    # (Input validation and data fetching logic unchanged)
    if not pathname or not pathname.startswith('/details/'): return html.Div(
        "Select a scenario's 'View Details' link from the Report page.")
    try:
        scenario_name = urllib.parse.unquote(pathname.split('/details/')[1])
    except IndexError:
        return html.Div("Invalid scenario name in URL.")
    scenario_inputs = stored_inputs.get(scenario_name)
    if not scenario_inputs: return html.Div(
        f"Could not find input data for scenario: {scenario_name}. Please add scenarios first.")
    corridor = scenario_inputs.get('corridor')
    disruption = scenario_inputs.get('disruption')
    if not corridor or not disruption: return html.Div(
        f"Missing context (corridor/disruption) for scenario: {scenario_name}")
    # --- Use the consistent baseline Re from archetype ---
    fitted_data, archetype, avg_ttr, avg_baseline_re_from_archetype, count, _, avg_adp_hours = run_historical_analysis(
        corridor, disruption)
    if archetype is None: return html.Div(f"Could not retrieve baseline data for {corridor}/{disruption}.")

    # (Helper function and 3-case analysis logic unchanged)
    def get_num_value(data_dict, key, default=0):
        val = data_dict.get(key)
        if val is None: return default
        try:
            num_val = float(val); return num_val if isinstance(num_val, (int, float)) else default
        except (ValueError, TypeError):
            return default

    results_list = []
    case_types = ["Optimistic", "Most Likely", "Pessimistic"]
    key_map = {"Optimistic": "min", "Most Likely": "mode", "Pessimistic": "max"}
    benefit_key_map = {"Optimistic": "max", "Most Likely": "mode", "Pessimistic": "min"}
    mode_params = {}
    for case in case_types:
        cost_key = key_map[case]
        benefit_key = benefit_key_map[case]
        # Get frequency for this case (min/mode/max)
        freq_val = get_num_value(scenario_inputs.get('disruption_frequency', {}), key_map[case],
                                 get_num_value(scenario_inputs.get('disruption_frequency', {}), 'mode'))
        annual_freq = None
        if freq_val is not None:
            try:
                annual_freq = float(freq_val) * 12  # Convert monthly to annual
            except (ValueError, TypeError):
                annual_freq = None

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
        if case == "Most Likely": mode_params = params.copy()  # mode_params now includes annual frequency

        # Pass the consistent baseline Re from archetype
        result = analyze_investment_scenario(None, archetype, avg_ttr, avg_adp_hours, financial_inputs, params,
                                             avg_baseline_re=avg_baseline_re_from_archetype)
        results_list.append(result)
    optimistic_result, likely_result, pessimistic_result = results_list
    baseline_roi = likely_result['business_case']['ROI']  # This is ROI relative to the archetype baseline now

    # --- Range Plot with background color & no gridlines ---
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
        # title="Scenario Performance Range", # Title removed as per request
        margin=dict(l=20, r=20, t=20, b=20),  # Adjusted margins
        template='plotly_white', legend_title_text='Cases', yaxis_tickformat='.0%',
        plot_bgcolor=COLOR_CARD_BACKGROUND, paper_bgcolor=COLOR_CARD_BACKGROUND,
        xaxis_showgrid=False, yaxis_showgrid=False  # Remove gridlines
    )

    # (Range Table logic unchanged)
    table_data = [optimistic_result['business_case'], likely_result['business_case'],
                  pessimistic_result['business_case']]
    range_df = pd.DataFrame(table_data)
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

    # (Sensitivity Analysis and Tornado Chart logic unchanged)
    # ... (omitted for brevity) ...
    sensitivity_data = []
    variables_to_test = [
        {"key": "implementation_cost", "label": "CAPEX", "type": "cost"},
        {"key": "annual_opex", "label": "OPEX", "type": "cost"},
        {"key": "perf_loss_reduction", "label": "Perf. Loss Reduction (%)", "type": "benefit"},
        {"key": "disruption_frequency", "label": "Monthly Frequency", "type": "cost"},
        # Example: Assuming higher freq = higher cost/lower ROI
        {"key": "new_adp_days", "label": "Adaptation Time (Days)", "type": "cost"},
        {"key": "new_ttr_days", "label": "TTR (Days)", "type": "cost"}
    ]
    for var in variables_to_test:
        var_key = var["key"]
        var_label = var["label"]
        var_type = var["type"]
        # Use mode_params[var_key] directly as the default if min/max not found
        mode_val_sens = get_num_value(scenario_inputs.get(var_key, {}), 'mode')  # Get mode value from original inputs
        if mode_val_sens is None: continue  # Skip if mode wasn't provided

        val_min_sens = get_num_value(scenario_inputs.get(var_key, {}), 'min', mode_val_sens)
        val_max_sens = get_num_value(scenario_inputs.get(var_key, {}), 'max', mode_val_sens)

        if val_min_sens == val_max_sens: continue  # Skip if no range provided

        # Recalculate Min scenario
        min_params_sens = mode_params.copy()  # Start with mode params (which has annual freq)
        min_params_sens[var_key] = val_min_sens
        # If the variable being tested IS frequency, convert min val to annual
        if var_key == "disruption_frequency":
            try:
                min_params_sens["frequency_per_year"] = float(val_min_sens) * 12
            except:
                min_params_sens["frequency_per_year"] = mode_params.get("frequency_per_year")  # Fallback

        min_result_sens = analyze_investment_scenario(None, archetype, avg_ttr, avg_adp_hours, financial_inputs,
                                                      min_params_sens, avg_baseline_re=avg_baseline_re_from_archetype)
        roi_at_min_input = min_result_sens['business_case']['ROI']

        # Recalculate Max scenario
        max_params_sens = mode_params.copy()
        max_params_sens[var_key] = val_max_sens
        # If the variable being tested IS frequency, convert max val to annual
        if var_key == "disruption_frequency":
            try:
                max_params_sens["frequency_per_year"] = float(val_max_sens) * 12
            except:
                max_params_sens["frequency_per_year"] = mode_params.get("frequency_per_year")  # Fallback

        max_result_sens = analyze_investment_scenario(None, archetype, avg_ttr, avg_adp_hours, financial_inputs,
                                                      max_params_sens, avg_baseline_re=avg_baseline_re_from_archetype)
        roi_at_max_input = max_result_sens['business_case']['ROI']

        low_roi_val, high_roi_val = (roi_at_max_input, roi_at_min_input) if var_type == 'cost' else (roi_at_min_input,
                                                                                                     roi_at_max_input)

        if not np.isfinite(low_roi_val): low_roi_val = -1000
        if not np.isfinite(high_roi_val): high_roi_val = 1000
        sensitivity_data.append({'label': var_label, 'low_roi': low_roi_val, 'high_roi': high_roi_val,
                                 'impact': abs(high_roi_val - low_roi_val)})

    # --- Tornado Chart with background color & no gridlines ---
    tornado_fig = go.Figure()
    if sensitivity_data:
        sensitivity_data.sort(key=lambda d: d['impact'], reverse=False)
        fig_data = []
        for d in sensitivity_data:
            finite_baseline_roi = baseline_roi if np.isfinite(baseline_roi) else 0
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
                template='plotly_white', margin=dict(l=150),
                plot_bgcolor=COLOR_CARD_BACKGROUND, paper_bgcolor=COLOR_CARD_BACKGROUND,
                xaxis_showgrid=False, yaxis_showgrid=False  # Remove gridlines
            )
            tornado_fig.add_vline(x=0, line_dash="dash", line_color="black")
        else:
            tornado_fig = go.Figure().update_layout(title="Sensitivity Analysis (No input ranges provided)",
                                                    plot_bgcolor=COLOR_CARD_BACKGROUND,
                                                    paper_bgcolor=COLOR_CARD_BACKGROUND, xaxis_showgrid=False,
                                                    yaxis_showgrid=False)  # Set background & no grid
    else:
        tornado_fig = go.Figure().update_layout(title="Sensitivity Analysis (No input ranges provided)",
                                                plot_bgcolor=COLOR_CARD_BACKGROUND, paper_bgcolor=COLOR_CARD_BACKGROUND,
                                                xaxis_showgrid=False, yaxis_showgrid=False)  # Set background & no grid

    # (Insights Card generation logic unchanged)
    insights_content = [html.H5("Key Insights", className="card-title"),
                        html.P("This chart shows which input variable has the biggest impact on your final ROI.")]
    if sensitivity_data:
        sensitivity_data.sort(key=lambda d: d['impact'], reverse=True)
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
                             className="shadow")  # Add style and shadow

    # --- Apply styling to cards in the details layout ---
    card_style = {'backgroundColor': COLOR_CARD_BACKGROUND, 'borderColor': COLOR_BORDER_LIGHT}
    card_class = "shadow"
    return html.Div([
        html.H1(f"Detailed Analysis: {scenario_name}", style={'color': 'black'}),  # Back to black
        html.Hr(),
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=fig_range))], style=card_style, className=card_class),
                    width=7),  # Header Removed
            dbc.Col(dbc.Card([dbc.CardHeader("Key Metric Ranges"), dbc.CardBody(
                dbc.Table.from_dataframe(range_df, striped=True, bordered=True, hover=True))], style=card_style,
                             className=card_class), width=5)
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(figure=tornado_fig))], style=card_style, className=card_class),
                    width=8),
            dbc.Col(insights_card, width=4)  # Style applied above
        ], className="mb-4"),
        html.Hr(),
        dbc.Button("Back to Report", href="/report", color="secondary")
    ])


# ==============================================================================
# 4. RUN THE APP
# ==============================================================================

if __name__ == '__main__':
    app.run(debug=True, port=8056)

