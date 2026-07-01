# app.py
# pyrefly: ignore [missing-import]
import os
import time

import streamlit as st
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import plotly.express as px
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go


from acceleration_engine import haversine_distance_gpu, optimize_routes, run_benchmark

# Concrete metrics (hardcoded from benchmark_concrete_metrics.py)
# GPU timing is measured with torch tensor broadcasting Haversine distance matrix.
# Note: CPU baseline is the naive nested-loop implementation; GPU numbers are the demo-critical ones.
CONCRETE_BENCHMARK = {
    "1000": {"gpu_time_s": 0.0226, "cpu_time_s": 8.6488},
    "2000": {"gpu_time_s": 0.0462, "cpu_time_s": 35.5370},
    "5000": {"gpu_time_s": 0.3606, "cpu_time_s": None},
    "10000": {"gpu_time_s": 1.5369, "cpu_time_s": None},
}

from gcp_connector import GCPConnector
from gemini_agent import DispatchAgent




# 1. Page Configuration
st.set_page_config(
    page_title="AFDRI - Accelerated Fleet Dispatch & Route Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 2. Theme Toggle Pattern
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

IS_DARK = st.session_state.theme == "dark"

# 3. CSS Design System Variables (dynamic light/dark values)
bg_val = "#000000" if IS_DARK else "#ffffff"
bg_subtle_val = "#050505" if IS_DARK else "#f9fafb"
card_val = "#0b0b0d" if IS_DARK else "#ffffff"
card_hover_val = "#121214" if IS_DARK else "#f4f4f5"
border_val = "#111114" if IS_DARK else "#e4e4e7"
border_subtle_val = "#141416" if IS_DARK else "#f0f0f2"
text_val = "#f8f8f8" if IS_DARK else "#000000"
text_muted_val = "rgba(248,248,248,0.85)" if IS_DARK else "#4b5563"
text_dim_val = "rgba(248,248,248,0.72)" if IS_DARK else "#6b7280"
shadow_val = "none" if IS_DARK else "0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.03)"
green_val = "#22c55e" if IS_DARK else "#16a34a"
green_muted_val = "rgba(34,197,94,0.22)" if IS_DARK else "rgba(22,163,74,0.08)"
red_val = "#ef4444" if IS_DARK else "#dc2626"
red_muted_val = "rgba(239,68,68,0.12)" if IS_DARK else "rgba(220,38,38,0.08)"
amber_val = "#f59e0b" if IS_DARK else "#d97706"
amber_muted_val = "rgba(245,158,11,0.12)" if IS_DARK else "rgba(217,119,6,0.08)"
accent_color = "#22c55e" if IS_DARK else "#2563eb"
accent_muted_color = "rgba(34,197,94,0.18)" if IS_DARK else "#1d4ed8"

# Read raw styles
with open("styles.css", "r") as f:
    css_styles = f.read()

# Inject dynamic CSS variables and layout rules
st.markdown(f"""
<style>
:root {{
    --bg: {bg_val};
    --bg-subtle: {bg_subtle_val};
    --card: {card_val};
    --card-hover: {card_hover_val};
    --border: {border_val};
    --border-subtle: {border_subtle_val};
    --text: {text_val};
    --text-muted: {text_muted_val};
    --text-dim: {text_dim_val};
    --accent: {accent_color};
    --accent-muted: {accent_muted_color};
    --green: {green_val};
    --green-muted: {green_muted_val};
    --red: {red_val};
    --red-muted: {red_muted_val};
    --amber: {amber_val};
    --amber-muted: {amber_muted_val};
    --shadow: {shadow_val};
    --radius: 10px;
}}
{css_styles}
</style>
""", unsafe_allow_html=True)

# 4. Styled Plotly Theme
plot_font_color = "#fafafa" if IS_DARK else "#09090b"
plot_tick_color = "#d4d4d8" if IS_DARK else "#525252"
plot_hover_bg = "rgba(20,20,20,0.9)" if IS_DARK else "rgba(255,255,255,0.9)"
plot_hover_font_color = "#fafafa" if IS_DARK else "#09090b"

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color=plot_font_color, size=11),
    margin=dict(l=20, r=20, t=30, b=20),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.04)" if IS_DARK else "rgba(0,0,0,0.04)",
        zerolinecolor="rgba(255,255,255,0.04)" if IS_DARK else "rgba(0,0,0,0.04)",
        tickfont=dict(size=10, color=plot_tick_color),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.04)" if IS_DARK else "rgba(0,0,0,0.04)",
        zerolinecolor="rgba(255,255,255,0.04)" if IS_DARK else "rgba(0,0,0,0.04)",
        tickfont=dict(size=10, color=plot_tick_color),
    ),
)

# Helpers
def metric_card(label, value, delta=None, delta_type="up"):
    cls = f"delta-{delta_type}"
    arrow = "↑" if delta_type == "up" else ("↓" if delta_type == "down" else "→")
    delta_html = f'<div class="metric-delta {cls}">{arrow} {delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

# 5. Header with brand + theme toggle
head_left, head_right = st.columns([8, 1.2])
with head_left:
    st.markdown("""
    <div class="brand">
        <span class="brand-icon">⚡</span>
        <span class="brand-name">AFDRI - Accelerated Fleet Dispatch & Route Intelligence</span>
    </div>
    """, unsafe_allow_html=True)
with head_right:
    theme_label = "☀️ Light Mode" if IS_DARK else "🌙 Dark Mode"
    st.button(theme_label, on_click=toggle_theme, width="stretch")

# 6. Initialize GCP Connector
if "gcp_conn" not in st.session_state:
    st.session_state.gcp_conn = GCPConnector()

connector = st.session_state.gcp_conn

# 7. Pills tabs navigation
tab_dispatch, tab_benchmarks, tab_gemini, tab_pipeline = st.tabs([
    "📍 Dispatch Control Center", 
    "📈 Performance Benchmarking", 
    "🤖 Gemini AI Assistant", 
    "📁 GCP Pipeline Flow"
])

# Maintain state for fleet and orders
if "fleet_df" not in st.session_state or "orders_df" not in st.session_state:
    st.session_state.fleet_df = connector.get_fleet_data()
    st.session_state.orders_df = connector.get_orders_data(250)
    st.session_state.assignments = {}
    st.session_state.unassigned = []
    st.session_state.calc_time = 0.0
    st.session_state.run_type = "None"

# TAB 1: Dispatch Console
with tab_dispatch:
    c_ctrls, c_map = st.columns([1, 2])
    
    with c_ctrls:
        st.subheader("Telemetry & Control")
        
        # Order Density Slider
        n_orders = st.slider("Simulated Active Orders (Density)", 50, 500, len(st.session_state.orders_df), step=50)
        if n_orders != len(st.session_state.orders_df):
            st.session_state.orders_df = connector.get_orders_data(n_orders)
            st.session_state.assignments = {}
            st.session_state.unassigned = []
            
        # Optimization Mode
        mode = st.radio("Optimization Mode", ["NVIDIA GPU (PyTorch CUDA)", "Standard CPU (Pandas Loop)"])
        
        if st.button("Optimize Route Dispatch", width="stretch", type="primary"):
            with st.spinner("Executing Distance Matrix & Greedy Assignment..."):
                orders = st.session_state.orders_df
                fleet = st.session_state.fleet_df
                
                # Check execution times
                start_t = time.perf_counter()
                
                # Pairwise Distance Matrix
                if "GPU" in mode:
                    dist_matrix = haversine_distance_gpu(orders['lat'].values, orders['lon'].values, use_cuda=True)
                    st.session_state.run_type = "GPU Accelerated"
                else:
                    # Slow CPU distance matrix computation
                    from acceleration_engine import haversine_distance_cpu_naive
                    dist_matrix = haversine_distance_cpu_naive(orders['lat'].values, orders['lon'].values)
                    st.session_state.run_type = "CPU Standard"
                    
                # Optimize / Greedy Route Assignment
                assignments, unassigned = optimize_routes(orders, fleet, dist_matrix)
                
                calc_time = time.perf_counter() - start_t
                st.session_state.assignments = assignments
                st.session_state.unassigned = unassigned
                st.session_state.calc_time = calc_time
                
                # Simulation Export to GCS
                route_records = []
                for v_id, o_idxs in assignments.items():
                    for step_idx, o_idx in enumerate(o_idxs):
                        ord_row = orders.iloc[o_idx]
                        route_records.append({
                            "vehicle_id": v_id,
                            "step": step_idx,
                            "order_id": ord_row["order_id"],
                            "lat": ord_row["lat"],
                            "lon": ord_row["lon"],
                            "demand": ord_row["demand"],
                            "priority": ord_row["priority"]
                        })
                if route_records:
                    export_uri = connector.export_routes_to_gcs(pd.DataFrame(route_records))
                    st.toast(f"Route logs written: {export_uri}", icon="📁")

        # KPI Metrics
        kpi_1, kpi_2 = st.columns(2)
        with kpi_1:
            metric_card(
                "Execution Time", 
                f"{st.session_state.calc_time:.4f}s", 
                delta="Faster" if "GPU" in st.session_state.run_type else "Normal",
                delta_type="up" if "GPU" in st.session_state.run_type else "warn"
            )
        with kpi_2:
            assigned_count = sum(len(x) for x in st.session_state.assignments.values())
            metric_card(
                "Capacity Utilization",
                f"{assigned_count} / {len(st.session_state.orders_df)}",
                delta=f"{len(st.session_state.unassigned)} Left",
                delta_type="down" if len(st.session_state.unassigned) > 0 else "up"
            )
            
    with c_map:
        st.subheader("Fleet Routing Visualization")
        
        # Build Plotly Map
        map_df = []
        orders_df = st.session_state.orders_df
        fleet_df = st.session_state.fleet_df
        assignments = st.session_state.assignments
        
        # Add all orders
        for idx, row in orders_df.iterrows():
            map_df.append({
                "lat": row["lat"],
                "lon": row["lon"],
                "label": f"Order {row['order_id']} (Demand: {row['demand']})",
                "color": "Orders",
                "size": 6
            })
            
        # Add all vehicles
        for idx, row in fleet_df.iterrows():
            map_df.append({
                "lat": row["lat"],
                "lon": row["lon"],
                "label": f"Driver {row['driver_name']} ({row['vehicle_id']})",
                "color": "Vehicles (Hubs)",
                "size": 12
            })
            
        m_df = pd.DataFrame(map_df)
        
        # If assignments are made, draw line paths
        fig = go.Figure()
        
        # Draw vehicle routes as connected line layers
        if assignments:
            colors = px.colors.qualitative.Plotly
            for v_idx, (v_id, o_idxs) in enumerate(assignments.items()):
                if not o_idxs:
                    continue
                v_row = fleet_df[fleet_df["vehicle_id"] == v_id].iloc[0]
                
                # Combine coordinates: Vehicle -> Order 1 -> Order 2 -> ...
                route_lats = [v_row["lat"]] + [orders_df.iloc[idx]["lat"] for idx in o_idxs]
                route_lons = [v_row["lon"]] + [orders_df.iloc[idx]["lon"] for idx in o_idxs]
                
                color = colors[v_idx % len(colors)]
                fig.add_trace(go.Scattermap(
                    lat=route_lats,
                    lon=route_lons,
                    mode="lines+markers",
                    line=dict(width=3, color=color),
                    marker=dict(size=8, color=color),
                    name=f"{v_row['driver_name']} ({v_id})",
                    hoverinfo="text",
                    text=[f"Start: {v_row['driver_name']}"] + [f"Order {orders_df.iloc[idx]['order_id']}" for idx in o_idxs]
                ))
        
        # Overlay orders and vehicle nodes
        fig.add_trace(go.Scattermap(
            lat=m_df[m_df["color"] == "Orders"]["lat"],
            lon=m_df[m_df["color"] == "Orders"]["lon"],
            mode="markers",
            marker=dict(size=6, color="#2563eb", opacity=0.8),
            name="Delivery Locations",
            text=m_df[m_df["color"] == "Orders"]["label"],
            hoverinfo="text",
            hoverlabel=dict(
                bgcolor=plot_hover_bg,
                bordercolor="rgba(255,255,255,0.1)",
                font=dict(color=plot_hover_font_color, family="DM Sans, sans-serif")
            )
        ))
        
        fig.add_trace(go.Scattermap(
            lat=m_df[m_df["color"] == "Vehicles (Hubs)"]["lat"],
            lon=m_df[m_df["color"] == "Vehicles (Hubs)"]["lon"],
            mode="markers",
            marker=dict(size=14, color="#ef4444", symbol="bus"),
            name="Driver Starts",
            text=m_df[m_df["color"] == "Vehicles (Hubs)"]["label"],
            hoverinfo="text",
            hoverlabel=dict(
                bgcolor=plot_hover_bg,
                bordercolor="rgba(255,255,255,0.1)",
                font=dict(color=plot_hover_font_color, family="DM Sans, sans-serif")
            )
        ))
        
        fig.update_layout(
            map=dict(
                style="open-street-map",
                center=dict(lat=40.73, lon=-73.97),
                zoom=10.5
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(9,9,11,0.85)" if IS_DARK else "rgba(255,255,255,0.85)",
                font=dict(color=plot_font_color)
            ),
            font=dict(color=plot_font_color),
            margin=dict(l=0, r=0, t=0, b=0),
            height=480
        )
        
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # Route Table Detail
    if assignments:
        st.subheader("Optimized Vehicle Dispatch Dispatch Schedule")
        rows_html = ""
        for v_id, o_idxs in assignments.items():
            driver_name = fleet_df[fleet_df["vehicle_id"] == v_id].iloc[0]["driver_name"]
            orders_assigned = ", ".join([orders_df.iloc[idx]["order_id"] for idx in o_idxs]) if o_idxs else "No Deliveries"
            total_orders = len(o_idxs)
            status_badge = '<span class="badge badge-green">Active Route</span>' if o_idxs else '<span class="badge badge-amber">Standby</span>'
            
            rows_html += f"""
            <tr>
                <td><b>{v_id}</b></td>
                <td>{driver_name}</td>
                <td>{status_badge}</td>
                <td>{total_orders}</td>
                <td>{orders_assigned}</td>
            </tr>
            """
        
        st.markdown(f"""
        <div class="table-wrap">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Vehicle ID</th>
                        <th>Driver</th>
                        <th>Status</th>
                        <th>Stop Count</th>
                        <th>Sequence Schedule</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

# TAB 2: Performance Benchmarks
with tab_benchmarks:
    st.subheader("GCP/NVIDIA GPU Acceleration Profiler")
    st.markdown(r"""
    In last-mile dispatch routing, computing the pairwise distance matrix scales at \(O(N^2)\). For 10,000 package delivery nodes, 
    the system must compute **100,000,000 (100 Million)** calculations. On standard Python/Pandas, this takes **minutes**, freezing real-time 
    dispatch. 
    
    Using **NVIDIA RAPIDS cuDF** or **PyTorch CUDA**, calculations run concurrently on the GPU core, reducing compute time to **milliseconds**.
    """)
    
    col_bench_run, col_bench_chart = st.columns([1, 2])
    
    with col_bench_run:
        st.markdown("### Run Live Profiler")
        st.write("Compare CPU Pandas processing against GPU-accelerated tensor math on your local NVIDIA RTX 3060.")
        
        bench_size_options = [100, 300, 500, 1000, 2000, 5000]
        select_all_checkbox = st.checkbox("Select all dataset sizes")
        selected_bench_options = st.multiselect(
            "Dataset sizes (nodes)",
            bench_size_options,
            default=bench_size_options if select_all_checkbox else [100, 300, 500, 1000],
        )
        bench_nodes = bench_size_options if select_all_checkbox else selected_bench_options
        
        if st.button("Execute Scale Benchmark", width="stretch", type="primary"):
            with st.spinner("Profiling calculations..."):
                bench_results = run_benchmark(bench_nodes)
                st.session_state.bench_data = bench_results
        if "bench_data" in st.session_state:
            # Display results summary table
            rows = ""
            for res in st.session_state.bench_data:
                cpu_t = f"{res['cpu_time']:.4f}s" if res['cpu_time'] is not None else "Skipped (>10s)"
                speedup_x = f"{res['speedup']:.1f}x"
                rows += f"""
                <tr>
                    <td><b>{res['size']}</b></td>
                    <td>{cpu_t}</td>
                    <td><span class="badge badge-green">{res['gpu_time']:.4f}s</span></td>
                    <td><span class="badge badge-blue">{speedup_x}</span></td>
                </tr>
                """
            st.markdown(f"""
            <div class="table-wrap">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Data Nodes (N)</th>
                            <th>CPU Pandas</th>
                            <th>NVIDIA GPU</th>
                            <th>Speedup Factor</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows}
                    </tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)
            
    with col_bench_chart:
        if "bench_data" in st.session_state:
            st.markdown("### Scaling and Time-to-Insight Curve")
            
            df_plot = pd.DataFrame(st.session_state.bench_data)
            
            # Line Chart comparing CPU vs GPU
            fig_bench = go.Figure()
            
            # Fill skipped CPU data for plotting projection
            cpu_plot_times = []
            for res in st.session_state.bench_data:
                if res['cpu_time'] is not None:
                    cpu_plot_times.append(res['cpu_time'])
                else:
                    # Projection
                    prev_n = 1000
                    prev_t = df_plot[df_plot['size'] == 1000]['cpu_time'].values[0]
                    proj_t = prev_t * ((res['size'] / prev_n) ** 2)
                    cpu_plot_times.append(proj_t)
            
            fig_bench.add_trace(go.Scatter(
                x=df_plot['size'], 
                y=cpu_plot_times,
                mode='lines+markers',
                name='CPU Pandas (Nested Loops)',
                line=dict(color='#ef4444', width=3)
            ))
            
            fig_bench.add_trace(go.Scatter(
                x=df_plot['size'], 
                y=df_plot['gpu_time'],
                mode='lines+markers',
                name='NVIDIA GPU (PyTorch/cuDF)',
                line=dict(color='#22c55e', width=3)
            ))
            
            PLOT_LAYOUT_BENCH = PLOT_LAYOUT.copy()
            PLOT_LAYOUT_BENCH.update(dict(
                xaxis_title="Number of Deliveries (N)",
                yaxis_title="Execution Time (Seconds)",
                height=350
            ))
            fig_bench.update_layout(PLOT_LAYOUT_BENCH)
            
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            st.plotly_chart(fig_bench, width="stretch", config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

# TAB 3: Gemini Chat Interface
with tab_gemini:
    st.subheader("Gemini Dispatch AI Assistant")
    st.write("Interact with your active dispatch database in natural language using Gemini Enterprise Agent Platform capability.")
    
    # Optional API key text input
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    api_key_input = st.text_input("Enter Gemini API Key (Optional - Simulated Mode runs automatically if left blank)", type="password")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Initialize Agent
    agent = DispatchAgent(api_key=api_key_input if api_key_input else None)
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    chat_container = st.container()
    
    # Render Chat History
    with chat_container:
        for sender, msg in st.session_state.chat_history:
            if sender == "User":
                st.markdown(f"🧑 **You**: {msg}")
            else:
                st.markdown(msg)
                
    st.markdown("---")
    
    user_query = st.chat_input("Ask about active routes (e.g., 'Analyze delay risks', 'Who is Sarah?')")
    if user_query:
        # Save message
        st.session_state.chat_history.append(("User", user_query))
        
        # Get AI response
        with st.spinner("Gemini reading database context..."):
            ans = agent.query_agent(
                user_query, 
                st.session_state.fleet_df, 
                st.session_state.orders_df, 
                st.session_state.assignments
            )
            st.session_state.chat_history.append(("AI", ans))
            st.rerun()

# TAB 4: Architecture
with tab_pipeline:
    st.subheader("AFDRI System Architecture & Pipelines")
    
    st.markdown("""
    This app showcases how GCP and NVIDIA cooperate at scale to power real-time supply chain decision tools:
    
    1. **BigQuery (Data Ingestion)**: Hosts static historic telemetry, order profiles, and geographical nodes.
    2. **Cloud Storage (Log Audits)**: Stores optimized dispatch plans (Parquet) for model training.
    3. **Managed Service for Apache Spark (NVIDIA RAPIDS Accelerator)**: Compiles routes on Spark RAPIDS using serverless NVIDIA GPUs on Google Cloud, processing millions of routes concurrently.
    4. **Gemini Enterprise Agent Platform**: Acts as natural language orchestrator over routing logs.
    """)
    
    # Graphical representation using Mermaid Diagram format
    st.markdown("""
    ```mermaid
    graph TD
        BQ[(BigQuery: Fleet telemetry & orders)] -->|Stream Data| Streamlit[Streamlit Control Center]
        Streamlit -->|Compute distance matrices| GPU[NVIDIA RTX 3060 / Spark RAPIDS GPU]
        GPU -->|Greedy Assignment| Streamlit
        Streamlit -->|Export routes logs as Parquet| GCS[(Google Cloud Storage Bucket)]
        Gemini[Gemini Dispatch AI Agent] -->|Queries & Context| Streamlit
    ```
    """)
