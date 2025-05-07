import streamlit as st
import subprocess
import os
import json
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import time
import io
import sys
import tempfile
import platform

# Set page config with wider layout
st.set_page_config(
    page_title="Matrix Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for a dashboard-like appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #f0f0f0;
    }
    .dashboard-container {
        background-color: #f9f9f9;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    .panel-header {
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #424242;
        border-left: 4px solid #1E88E5;
        padding-left: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f0f0;
        border-radius: 6px 6px 0 0;
        gap: 1;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5 !important;
        color: white !important;
    }
    .math-box {
        background-color: #f8f9fa;
        border-left: 3px solid #1E88E5;
        padding: 10px;
        margin: 10px 0;
    }
    .stWarning {
        background-color: #fff3cd;
        padding: 10px;
        border-left: 3px solid #ffc107;
        margin: 10px 0;
    }
    .stSuccess {
        background-color: #d4edda;
        padding: 10px;
        border-left: 3px solid #28a745;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Dashboard Header
st.markdown('<h1 class="main-header">Matrix Analysis Dashboard</h1>', unsafe_allow_html=True)

# Create output directory in the current working directory
current_dir = os.getcwd()
output_dir = os.path.join(current_dir, "output")
os.makedirs(output_dir, exist_ok=True)

# Path to the C++ source file and executable
cpp_file = os.path.join(current_dir, "app.cpp")
executable = os.path.join(current_dir, "eigen_analysis")
if platform.system() == "Windows":
    executable += ".exe"

# Helper function for running commands with better debugging
def run_command(cmd, show_output=True, timeout=None):
    cmd_str = " ".join(cmd)
    if show_output:
        st.code(f"Running command: {cmd_str}", language="bash")
    
    # Run the command
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=timeout
        )
        
        if result.returncode == 0:
            if show_output:
                st.success("Command completed successfully.")
                if result.stdout and show_output:
                    with st.expander("Command Output"):
                        st.code(result.stdout)
            return True, result.stdout, result.stderr
        else:
            if show_output:
                st.error(f"Command failed with return code {result.returncode}")
                st.error(f"Command: {cmd_str}")
                st.error(f"Error output: {result.stderr}")
            return False, result.stdout, result.stderr
    
    except subprocess.TimeoutExpired:
        if show_output:
            st.error(f"Command timed out after {timeout} seconds")
        return False, "", f"Command timed out after {timeout} seconds"
    except Exception as e:
        if show_output:
            st.error(f"Error executing command: {str(e)}")
        return False, "", str(e)

# Check if C++ source file exists
if not os.path.exists(cpp_file):
    with open(cpp_file, "w") as f:
        st.warning(f"C++ source file not found at: {cpp_file}")
        st.info("Creating an empty file. Please paste the C++ code into this file and recompile.")
        f.write("// Paste the C++ code here and recompile\n")

# Compile the C++ code with the right OpenCV libraries
st.sidebar.title("Compiler Settings")
need_compile = not os.path.exists(executable) or st.sidebar.button("Recompile C++ Code")

if need_compile:
    with st.sidebar:
        with st.spinner("Compiling C++ code..."):
            # Try to detect the OpenCV installation
            opencv_detection_cmd = ["pkg-config", "--cflags", "--libs", "opencv4"]
            opencv_found, opencv_flags, _ = run_command(opencv_detection_cmd, show_output=False)
            
            compile_commands = []
            
            if opencv_found:
                compile_commands.append(
                    f"g++ -o {executable} {cpp_file} {opencv_flags.strip()} -std=c++11"
                )
            else:
                # Try different OpenCV configurations
                compile_commands = [
                    f"g++ -o {executable} {cpp_file} `pkg-config --cflags --libs opencv4` -std=c++11",
                    f"g++ -o {executable} {cpp_file} `pkg-config --cflags --libs opencv` -std=c++11",
                    f"g++ -o {executable} {cpp_file} -I/usr/include/opencv4 -lopencv_core -lopencv_imgproc -std=c++11",
                    f"g++ -o {executable} {cpp_file} -I/usr/local/include/opencv4 -lopencv_core -lopencv_imgproc -std=c++11"
                ]
            
            compiled = False
            compile_output = ""
            
            for cmd in compile_commands:
                st.text(f"Trying: {cmd}")
                success, stdout, stderr = run_command(cmd.split(), show_output=False)
                compile_output += f"Command: {cmd}\nOutput: {stdout}\nError: {stderr}\n\n"
                
                if success:
                    compiled = True
                    st.success(f"Successfully compiled with: {cmd}")
                    break
            
            if not compiled:
                st.error("All compilation attempts failed.")
                with st.expander("Compilation Details"):
                    st.code(compile_output)
                st.stop()
            
            # Make sure the executable is executable
            if platform.system() != "Windows":
                os.chmod(executable, 0o755)
            
            st.success("C++ code compiled successfully!")

# Create tabs for different analyses
tab1, tab2 = st.tabs(["Eigenvalue Analysis", "Im(s) vs z Analysis"])

# Tab 1: Eigenvalue Analysis
with tab1:
    # Two-column layout for the dashboard
    left_column, right_column = st.columns([1, 3])
    
    with left_column:
        st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">Eigenvalue Analysis Controls</div>', unsafe_allow_html=True)
        
        # Parameter inputs with defaults and validation
        st.markdown("### Matrix Parameters")
        n = st.number_input("Sample size (n)", min_value=5, max_value=1000, value=100, step=5, 
                           help="Number of samples", key="eig_n")
        p = st.number_input("Dimension (p)", min_value=5, max_value=1000, value=50, step=5, 
                           help="Dimensionality", key="eig_p")
        a = st.number_input("Value for a", min_value=1.1, max_value=10.0, value=2.0, step=0.1, 
                           help="Parameter a > 1", key="eig_a")
        
        # Automatically calculate y = p/n (as requested)
        y = p/n
        st.info(f"Value for y = p/n: {y:.4f}")
        
        st.markdown("### Calculation Controls")
        fineness = st.slider(
            "Beta points", 
            min_value=20, 
            max_value=500, 
            value=100, 
            step=10,
            help="Number of points to calculate along the β axis (0 to 1)",
            key="eig_fineness"
        )
        
        with st.expander("Advanced Settings"):
            # Add controls for theoretical calculation precision
            theory_grid_points = st.slider(
                "Theoretical grid points", 
                min_value=100, 
                max_value=1000, 
                value=200, 
                step=50,
                help="Number of points in initial grid search for theoretical calculations",
                key="eig_grid_points"
            )
            
            theory_tolerance = st.number_input(
                "Theoretical tolerance", 
                min_value=1e-12, 
                max_value=1e-6, 
                value=1e-10, 
                format="%.1e",
                help="Convergence tolerance for golden section search",
                key="eig_tolerance"
            )
            
            # Debug mode
            debug_mode = st.checkbox("Debug Mode", value=False, key="eig_debug")
            
            # Timeout setting
            timeout_seconds = st.number_input(
                "Computation timeout (seconds)", 
                min_value=30, 
                max_value=3600, 
                value=300,
                help="Maximum time allowed for computation before timeout",
                key="eig_timeout"
            )
        
        # Generate button
        eig_generate_button = st.button("Generate Eigenvalue Analysis", 
                                      type="primary", 
                                      use_container_width=True,
                                      key="eig_generate")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with right_column:
        # Main visualization area
        st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">Eigenvalue Analysis Results</div>', unsafe_allow_html=True)
        
        # Container for the analysis results
        eig_results_container = st.container()
        
        # Process when generate button is clicked
        if eig_generate_button:
            with eig_results_container:
                # Show progress
                progress_container = st.container()
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                try:
                    # Create data file path
                    data_file = os.path.join(output_dir, "eigenvalue_data.json")
                    
                    # Delete previous output if exists
                    if os.path.exists(data_file):
                        os.remove(data_file)
                    
                    # Build command for eigenvalue analysis with the proper arguments
                    cmd = [
                        executable,
                        "eigenvalues",  # Mode argument
                        str(n),
                        str(p),
                        str(a),
                        str(y),
                        str(fineness),
                        str(theory_grid_points),
                        str(theory_tolerance),
                        data_file
                    ]
                    
                    # Run the command
                    status_text.text("Running eigenvalue analysis...")
                    
                    if debug_mode:
                        success, stdout, stderr = run_command(cmd, True, timeout=timeout_seconds)
                        # Process stdout for progress updates
                        if success:
                            progress_bar.progress(1.0)
                    else:
                        # Start the process with pipe for stdout to read progress
                        process = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            bufsize=1,
                            universal_newlines=True
                        )
                        
                        # Track progress from stdout
                        success = True
                        stdout_lines = []
                        
                        start_time = time.time()
                        while True:
                            # Check for timeout
                            if time.time() - start_time > timeout_seconds:
                                process.kill()
                                status_text.error(f"Computation timed out after {timeout_seconds} seconds")
                                success = False
                                break
                                
                            # Try to read a line (non-blocking)
                            line = process.stdout.readline()
                            if not line and process.poll() is not None:
                                break
                                
                            if line:
                                stdout_lines.append(line)
                                if line.startswith("PROGRESS:"):
                                    try:
                                        # Update progress bar
                                        progress_value = float(line.split(":")[1].strip())
                                        progress_bar.progress(progress_value)
                                        status_text.text(f"Calculating... {int(progress_value * 100)}% complete")
                                    except:
                                        pass
                                elif line:
                                    status_text.text(line.strip())
                                    
                        # Get the return code and stderr
                        returncode = process.poll()
                        stderr = process.stderr.read()
                        
                        if returncode != 0:
                            success = False
                            st.error(f"Error executing the analysis: {stderr}")
                            with st.expander("Error Details"):
                                st.code(stderr)
                    
                    if success:
                        progress_bar.progress(1.0)
                        status_text.text("Analysis complete! Generating visualization...")
                        
                        # Check if the output file was created
                        if not os.path.exists(data_file):
                            st.error(f"Output file not created: {data_file}")
                            st.stop()
                        
                        try:
                            # Load the results from the JSON file
                            with open(data_file, 'r') as f:
                                data = json.load(f)
                            
                            # Extract data
                            beta_values = np.array(data['beta_values'])
                            max_eigenvalues = np.array(data['max_eigenvalues'])
                            min_eigenvalues = np.array(data['min_eigenvalues'])
                            theoretical_max = np.array(data['theoretical_max'])
                            theoretical_min = np.array(data['theoretical_min'])
                            
                            # Create an interactive plot using Plotly
                            fig = go.Figure()
                            
                            # Add traces for each line
                            fig.add_trace(go.Scatter(
                                x=beta_values, 
                                y=max_eigenvalues,
                                mode='lines+markers',
                                name='Empirical Max Eigenvalue',
                                line=dict(color='rgb(220, 60, 60)', width=3),
                                marker=dict(
                                    symbol='circle',
                                    size=8,
                                    color='rgb(220, 60, 60)',
                                    line=dict(color='white', width=1)
                                ),
                                hovertemplate='β: %{x:.3f}<br>Value: %{y:.6f}<extra>Empirical Max</extra>'
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=beta_values, 
                                y=min_eigenvalues,
                                mode='lines+markers',
                                name='Empirical Min Eigenvalue',
                                line=dict(color='rgb(60, 60, 220)', width=3),
                                marker=dict(
                                    symbol='circle',
                                    size=8,
                                    color='rgb(60, 60, 220)',
                                    line=dict(color='white', width=1)
                                ),
                                hovertemplate='β: %{x:.3f}<br>Value: %{y:.6f}<extra>Empirical Min</extra>'
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=beta_values, 
                                y=theoretical_max,
                                mode='lines+markers',
                                name='Theoretical Max Function',
                                line=dict(color='rgb(30, 180, 30)', width=3),
                                marker=dict(
                                    symbol='diamond',
                                    size=8,
                                    color='rgb(30, 180, 30)',
                                    line=dict(color='white', width=1)
                                ),
                                hovertemplate='β: %{x:.3f}<br>Value: %{y:.6f}<extra>Theoretical Max</extra>'
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=beta_values, 
                                y=theoretical_min,
                                mode='lines+markers',
                                name='Theoretical Min Function',
                                line=dict(color='rgb(180, 30, 180)', width=3),
                                marker=dict(
                                    symbol='diamond',
                                    size=8,
                                    color='rgb(180, 30, 180)',
                                    line=dict(color='white', width=1)
                                ),
                                hovertemplate='β: %{x:.3f}<br>Value: %{y:.6f}<extra>Theoretical Min</extra>'
                            ))
                            
                            # Configure layout for better appearance
                            fig.update_layout(
                                title={
                                    'text': f'Eigenvalue Analysis: n={n}, p={p}, a={a}, y={y:.4f}',
                                    'font': {'size': 24, 'color': '#1E88E5'},
                                    'y': 0.95,
                                    'x': 0.5,
                                    'xanchor': 'center',
                                    'yanchor': 'top'
                                },
                                xaxis={
                                    'title': {'text': 'β Parameter', 'font': {'size': 18, 'color': '#424242'}},
                                    'tickfont': {'size': 14},
                                    'gridcolor': 'rgba(220, 220, 220, 0.5)',
                                    'showgrid': True
                                },
                                yaxis={
                                    'title': {'text': 'Eigenvalues', 'font': {'size': 18, 'color': '#424242'}},
                                    'tickfont': {'size': 14},
                                    'gridcolor': 'rgba(220, 220, 220, 0.5)',
                                    'showgrid': True
                                },
                                plot_bgcolor='rgba(240, 240, 240, 0.8)',
                                paper_bgcolor='rgba(249, 249, 249, 0.8)',
                                hovermode='closest',
                                legend={
                                    'font': {'size': 14},
                                    'bgcolor': 'rgba(255, 255, 255, 0.9)',
                                    'bordercolor': 'rgba(200, 200, 200, 0.5)',
                                    'borderwidth': 1
                                },
                                margin={'l': 60, 'r': 30, 't': 100, 'b': 60},
                                height=600,
                                annotations=[
                                    {
                                        'text': f"Max Function: max{{k ∈ (0,∞)}} [yβ(a-1)k + (ak+1)((y-1)k-1)]/[(ak+1)(k²+k)]",
                                        'xref': 'paper', 'yref': 'paper',
                                        'x': 0.02, 'y': 0.02,
                                        'showarrow': False,
                                        'font': {'size': 12, 'color': 'rgb(30, 180, 30)'},
                                        'bgcolor': 'rgba(255, 255, 255, 0.9)',
                                        'bordercolor': 'rgb(30, 180, 30)',
                                        'borderwidth': 1,
                                        'borderpad': 4
                                    },
                                    {
                                        'text': f"Min Function: min{{t ∈ (-1/a,0)}} [yβ(a-1)t + (at+1)((y-1)t-1)]/[(at+1)(t²+t)]",
                                        'xref': 'paper', 'yref': 'paper',
                                        'x': 0.55, 'y': 0.02,
                                        'showarrow': False,
                                        'font': {'size': 12, 'color': 'rgb(180, 30, 180)'},
                                        'bgcolor': 'rgba(255, 255, 255, 0.9)',
                                        'bordercolor': 'rgb(180, 30, 180)',
                                        'borderwidth': 1,
                                        'borderpad': 4
                                    }
                                ]
                            )
                            
                            # Add custom modebar buttons
                            fig.update_layout(
                                modebar_add=[
                                    'drawline', 'drawopenpath', 'drawclosedpath',
                                    'drawcircle', 'drawrect', 'eraseshape'
                                ],
                                modebar_remove=['lasso2d', 'select2d'],
                                dragmode='zoom'
                            )
                            
                            # Clear progress container
                            progress_container.empty()
                            
                            # Display the interactive plot in Streamlit
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display statistics
                            with st.expander("Statistics"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("### Eigenvalue Statistics")
                                    st.write(f"Max empirical value: {max_eigenvalues.max():.6f}")
                                    st.write(f"Min empirical value: {min_eigenvalues.min():.6f}")
                                with col2:
                                    st.write("### Theoretical Values")
                                    st.write(f"Max theoretical value: {theoretical_max.max():.6f}")
                                    st.write(f"Min theoretical value: {theoretical_min.min():.6f}")
                                
                        except json.JSONDecodeError as e:
                            st.error(f"Error parsing JSON results: {str(e)}")
                            if os.path.exists(data_file):
                                with open(data_file, 'r') as f:
                                    content = f.read()
                                st.code(content[:1000] + "..." if len(content) > 1000 else content)
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    if debug_mode:
                        st.exception(e)
        
        else:
            # Try to load existing data if available
            data_file = os.path.join(output_dir, "eigenvalue_data.json")
            if os.path.exists(data_file):
                try:
                    with open(data_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract data
                    beta_values = np.array(data['beta_values'])
                    max_eigenvalues = np.array(data['max_eigenvalues'])
                    min_eigenvalues = np.array(data['min_eigenvalues'])
                    theoretical_max = np.array(data['theoretical_max'])
                    theoretical_min = np.array(data['theoretical_min'])
                    
                    # Create an interactive plot using Plotly
                    fig = go.Figure()
                    
                    # Add traces for each line
                    fig.add_trace(go.Scatter(
                        x=beta_values, 
                        y=max_eigenvalues,
                        mode='lines+markers',
                        name='Empirical Max Eigenvalue',
                        line=dict(color='rgb(220, 60, 60)', width=3),
                        marker=dict(
                            symbol='circle',
                            size=8,
                            color='rgb(220, 60, 60)',
                            line=dict(color='white', width=1)
                        ),
                        hovertemplate='β: %{x:.3f}<br>Value: %{y:.6f}<extra>Empirical Max</extra>'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=beta_values, 
                        y=min_eigenvalues,
                        mode='lines+markers',
                        name='Empirical Min Eigenvalue',
                        line=dict(color='rgb(60, 60, 220)', width=3),
                        marker=dict(
                            symbol='circle',
                            size=8,
                            color='rgb(60, 60, 220)',
                            line=dict(color='white', width=1)
                        ),
                        hovertemplate='β: %{x:.3f}<br>Value: %{y:.6f}<extra>Empirical Min</extra>'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=beta_values, 
                        y=theoretical_max,
                        mode='lines+markers',
                        name='Theoretical Max Function',
                        line=dict(color='rgb(30, 180, 30)', width=3),
                        marker=dict(
                            symbol='diamond',
                            size=8,
                            color='rgb(30, 180, 30)',
                            line=dict(color='white', width=1)
                        ),
                        hovertemplate='β: %{x:.3f}<br>Value: %{y:.6f}<extra>Theoretical Max</extra>'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=beta_values, 
                        y=theoretical_min,
                        mode='lines+markers',
                        name='Theoretical Min Function',
                        line=dict(color='rgb(180, 30, 180)', width=3),
                        marker=dict(
                            symbol='diamond',
                            size=8,
                            color='rgb(180, 30, 180)',
                            line=dict(color='white', width=1)
                        ),
                        hovertemplate='β: %{x:.3f}<br>Value: %{y:.6f}<extra>Theoretical Min</extra>'
                    ))
                    
                    # Configure layout for better appearance
                    fig.update_layout(
                        title={
                            'text': f'Eigenvalue Analysis (Previous Result)',
                            'font': {'size': 24, 'color': '#1E88E5'},
                            'y': 0.95,
                            'x': 0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'
                        },
                        xaxis={
                            'title': {'text': 'β Parameter', 'font': {'size': 18, 'color': '#424242'}},
                            'tickfont': {'size': 14},
                            'gridcolor': 'rgba(220, 220, 220, 0.5)',
                            'showgrid': True
                        },
                        yaxis={
                            'title': {'text': 'Eigenvalues', 'font': {'size': 18, 'color': '#424242'}},
                            'tickfont': {'size': 14},
                            'gridcolor': 'rgba(220, 220, 220, 0.5)',
                            'showgrid': True
                        },
                        plot_bgcolor='rgba(240, 240, 240, 0.8)',
                        paper_bgcolor='rgba(249, 249, 249, 0.8)',
                        hovermode='closest',
                        legend={
                            'font': {'size': 14},
                            'bgcolor': 'rgba(255, 255, 255, 0.9)',
                            'bordercolor': 'rgba(200, 200, 200, 0.5)',
                            'borderwidth': 1
                        },
                        margin={'l': 60, 'r': 30, 't': 100, 'b': 60},
                        height=600
                    )
                    
                    # Display the interactive plot in Streamlit
                    st.plotly_chart(fig, use_container_width=True)
                    st.info("This is the previous analysis result. Adjust parameters and click 'Generate Analysis' to create a new visualization.")
                    
                except Exception as e:
                    st.info("👈 Set parameters and click 'Generate Eigenvalue Analysis' to create a visualization.")
            else:
                # Show placeholder
                st.info("👈 Set parameters and click 'Generate Eigenvalue Analysis' to create a visualization.")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: Im(s) vs z Analysis
with tab2:
    # Two-column layout for the dashboard
    left_column, right_column = st.columns([1, 3])
    
    with left_column:
        st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">Im(s) vs z Analysis Controls</div>', unsafe_allow_html=True)
        
        # Parameter inputs with defaults and validation
        st.markdown("### Cubic Equation Parameters")
        cubic_a = st.number_input("Value for a", min_value=1.1, max_value=10.0, value=2.0, step=0.1, 
                                help="Parameter a > 1", key="cubic_a")
        cubic_y = st.number_input("Value for y", min_value=0.1, max_value=10.0, value=1.0, step=0.1,
                                 help="Parameter y > 0", key="cubic_y")
        cubic_beta = st.number_input("Value for β", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                                   help="Value between 0 and 1", key="cubic_beta")
        
        st.markdown("### Calculation Controls")
        cubic_points = st.slider(
            "Number of z points", 
            min_value=50, 
            max_value=1000, 
            value=300, 
            step=50,
            help="Number of points to calculate along the z axis",
            key="cubic_points"
        )
        
        # Debug mode
        cubic_debug_mode = st.checkbox("Debug Mode", value=False, key="cubic_debug")
        
        # Timeout setting
        cubic_timeout = st.number_input(
            "Computation timeout (seconds)", 
            min_value=10, 
            max_value=600, 
            value=60,
            help="Maximum time allowed for computation before timeout",
            key="cubic_timeout"
        )
        
        # Show cubic equation
        st.markdown('<div class="math-box">', unsafe_allow_html=True)
        st.markdown("### Cubic Equation")
        st.latex(r"zas^3 + [z(a+1)+a(1-y)]\,s^2 + [z+(a+1)-y-y\beta (a-1)]\,s + 1 = 0")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Generate button
        cubic_generate_button = st.button("Generate Im(s) vs z Analysis", 
                                        type="primary", 
                                        use_container_width=True,
                                        key="cubic_generate")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with right_column:
        # Main visualization area
        st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">Im(s) vs z Analysis Results</div>', unsafe_allow_html=True)
        
        # Container for the analysis results
        cubic_results_container = st.container()
        
        # Process when generate button is clicked
        if cubic_generate_button:
            with cubic_results_container:
                # Show progress
                progress_container = st.container()
                with progress_container:
                    status_text = st.empty()
                    status_text.text("Starting cubic equation calculations...")
                
                try:
                    # Run the C++ executable with the parameters in JSON output mode
                    data_file = os.path.join(output_dir, "cubic_data.json")
                    
                    # Delete previous output if exists
                    if os.path.exists(data_file):
                        os.remove(data_file)
                    
                    # Build command for cubic equation analysis
                    cmd = [
                        executable,
                        "cubic",  # Mode argument
                        str(cubic_a),
                        str(cubic_y),
                        str(cubic_beta),
                        str(cubic_points),
                        data_file
                    ]
                    
                    # Run the command
                    status_text.text("Calculating Im(s) vs z values...")
                    
                    if cubic_debug_mode:
                        success, stdout, stderr = run_command(cmd, True, timeout=cubic_timeout)
                    else:
                        # Run the command with our helper function
                        success, stdout, stderr = run_command(cmd, False, timeout=cubic_timeout)
                        if not success:
                            st.error(f"Error executing cubic analysis: {stderr}")
                    
                    if success:
                        status_text.text("Calculations complete! Generating visualization...")
                        
                        # Check if the output file was created
                        if not os.path.exists(data_file):
                            st.error(f"Output file not created: {data_file}")
                            st.stop()
                        
                        try:
                            # Load the results from the JSON file
                            with open(data_file, 'r') as f:
                                data = json.load(f)
                            
                            # Extract data
                            z_values = np.array(data['z_values'])
                            ims_values1 = np.array(data['ims_values1'])
                            ims_values2 = np.array(data['ims_values2'])
                            ims_values3 = np.array(data['ims_values3'])
                            
                            # Create an interactive plot using Plotly
                            fig = go.Figure()
                            
                            # Add traces for each root's imaginary part
                            fig.add_trace(go.Scatter(
                                x=z_values, 
                                y=ims_values1,
                                mode='lines',
                                name='Im(s₁)',
                                line=dict(color='rgb(220, 60, 60)', width=3),
                                hovertemplate='z: %{x:.3f}<br>Im(s₁): %{y:.6f}<extra>Root 1</extra>'
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=z_values, 
                                y=ims_values2,
                                mode='lines',
                                name='Im(s₂)',
                                line=dict(color='rgb(60, 60, 220)', width=3),
                                hovertemplate='z: %{x:.3f}<br>Im(s₂): %{y:.6f}<extra>Root 2</extra>'
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=z_values, 
                                y=ims_values3,
                                mode='lines',
                                name='Im(s₃)',
                                line=dict(color='rgb(30, 180, 30)', width=3),
                                hovertemplate='z: %{x:.3f}<br>Im(s₃): %{y:.6f}<extra>Root 3</extra>'
                            ))
                            
                            # Configure layout for better appearance
                            fig.update_layout(
                                title={
                                    'text': f'Im(s) vs z Analysis: a={cubic_a}, y={cubic_y}, β={cubic_beta}',
                                    'font': {'size': 24, 'color': '#1E88E5'},
                                    'y': 0.95,
                                    'x': 0.5,
                                    'xanchor': 'center',
                                    'yanchor': 'top'
                                },
                                xaxis={
                                    'title': {'text': 'z (logarithmic scale)', 'font': {'size': 18, 'color': '#424242'}},
                                    'tickfont': {'size': 14},
                                    'gridcolor': 'rgba(220, 220, 220, 0.5)',
                                    'showgrid': True,
                                    'type': 'log'  # Use logarithmic scale for better visualization
                                },
                                yaxis={
                                    'title': {'text': 'Im(s)', 'font': {'size': 18, 'color': '#424242'}},
                                    'tickfont': {'size': 14},
                                    'gridcolor': 'rgba(220, 220, 220, 0.5)',
                                    'showgrid': True
                                },
                                plot_bgcolor='rgba(240, 240, 240, 0.8)',
                                paper_bgcolor='rgba(249, 249, 249, 0.8)',
                                hovermode='closest',
                                legend={
                                    'font': {'size': 14},
                                    'bgcolor': 'rgba(255, 255, 255, 0.9)',
                                    'bordercolor': 'rgba(200, 200, 200, 0.5)',
                                    'borderwidth': 1
                                },
                                margin={'l': 60, 'r': 30, 't': 100, 'b': 60},
                                height=600,
                                annotations=[
                                    {
                                        'text': f"Cubic Equation: {cubic_a}zs³ + [{cubic_a+1}z+{cubic_a}(1-{cubic_y})]s² + [z+{cubic_a+1}-{cubic_y}-{cubic_y*cubic_beta}({cubic_a-1})]s + 1 = 0",
                                        'xref': 'paper', 'yref': 'paper',
                                        'x': 0.5, 'y': 0.02,
                                        'showarrow': False,
                                        'font': {'size': 12, 'color': 'black'},
                                        'bgcolor': 'rgba(255, 255, 255, 0.9)',
                                        'bordercolor': 'rgba(0, 0, 0, 0.5)',
                                        'borderwidth': 1,
                                        'borderpad': 4,
                                        'align': 'center'
                                    }
                                ]
                            )
                            
                            # Add custom modebar buttons
                            fig.update_layout(
                                modebar_add=[
                                    'drawline', 'drawopenpath', 'drawclosedpath',
                                    'drawcircle', 'drawrect', 'eraseshape'
                                ],
                                modebar_remove=['lasso2d', 'select2d'],
                                dragmode='zoom'
                            )
                            
                            # Clear progress container
                            progress_container.empty()
                            
                            # Display the interactive plot in Streamlit
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add explanation text
                            st.markdown("""
                            ### Explanation of the Analysis
                            
                            This plot shows the imaginary parts of the three roots (s₁, s₂, s₃) of the cubic equation as a function of z. 
                            The cubic equation being solved is:
                            
                            ```
                            zas³ + [z(a+1)+a(1-y)]s² + [z+(a+1)-y-yβ(a-1)]s + 1 = 0
                            ```
                            
                            Where a, y, and β are parameters you can adjust in the control panel. The imaginary parts of the roots represent 
                            oscillatory behavior in the system.
                            
                            - When Im(s) = 0, the root is purely real
                            - When Im(s) ≠ 0, the root has an oscillatory component
                            """)
                            
                        except json.JSONDecodeError as e:
                            st.error(f"Error parsing JSON results: {str(e)}")
                            if os.path.exists(data_file):
                                with open(data_file, 'r') as f:
                                    content = f.read()
                                st.code(content[:1000] + "..." if len(content) > 1000 else content)
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    if cubic_debug_mode:
                        st.exception(e)
        
        else:
            # Try to load existing data if available
            data_file = os.path.join(output_dir, "cubic_data.json")
            if os.path.exists(data_file):
                try:
                    with open(data_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract data
                    z_values = np.array(data['z_values'])
                    ims_values1 = np.array(data['ims_values1'])
                    ims_values2 = np.array(data['ims_values2'])
                    ims_values3 = np.array(data['ims_values3'])
                    
                    # Create an interactive plot using Plotly
                    fig = go.Figure()
                    
                    # Add traces for each root's imaginary part
                    fig.add_trace(go.Scatter(
                        x=z_values, 
                        y=ims_values1,
                        mode='lines',
                        name='Im(s₁)',
                        line=dict(color='rgb(220, 60, 60)', width=3),
                        hovertemplate='z: %{x:.3f}<br>Im(s₁): %{y:.6f}<extra>Root 1</extra>'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=z_values, 
                        y=ims_values2,
                        mode='lines',
                        name='Im(s₂)',
                        line=dict(color='rgb(60, 60, 220)', width=3),
                        hovertemplate='z: %{x:.3f}<br>Im(s₂): %{y:.6f}<extra>Root 2</extra>'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=z_values, 
                        y=ims_values3,
                        mode='lines',
                        name='Im(s₃)',
                        line=dict(color='rgb(30, 180, 30)', width=3),
                        hovertemplate='z: %{x:.3f}<br>Im(s₃): %{y:.6f}<extra>Root 3</extra>'
                    ))
                    
                    # Configure layout for better appearance
                    fig.update_layout(
                        title={
                            'text': f'Im(s) vs z Analysis (Previous Result)',
                            'font': {'size': 24, 'color': '#1E88E5'},
                            'y': 0.95,
                            'x': 0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'
                        },
                        xaxis={
                            'title': {'text': 'z (logarithmic scale)', 'font': {'size': 18, 'color': '#424242'}},
                            'tickfont': {'size': 14},
                            'gridcolor': 'rgba(220, 220, 220, 0.5)',
                            'showgrid': True,
                            'type': 'log'  # Use logarithmic scale for better visualization
                        },
                        yaxis={
                            'title': {'text': 'Im(s)', 'font': {'size': 18, 'color': '#424242'}},
                            'tickfont': {'size': 14},
                            'gridcolor': 'rgba(220, 220, 220, 0.5)',
                            'showgrid': True
                        },
                        plot_bgcolor='rgba(240, 240, 240, 0.8)',
                        paper_bgcolor='rgba(249, 249, 249, 0.8)',
                        hovermode='closest',
                        legend={
                            'font': {'size': 14},
                            'bgcolor': 'rgba(255, 255, 255, 0.9)',
                            'bordercolor': 'rgba(200, 200, 200, 0.5)',
                            'borderwidth': 1
                        },
                        margin={'l': 60, 'r': 30, 't': 100, 'b': 60},
                        height=600
                    )
                    
                    # Display the interactive plot in Streamlit
                    st.plotly_chart(fig, use_container_width=True)
                    st.info("This is the previous analysis result. Adjust parameters and click 'Generate Analysis' to create a new visualization.")
                    
                except Exception as e:
                    st.info("👈 Set parameters and click 'Generate Im(s) vs z Analysis' to create a visualization.")
            else:
                # Show placeholder
                st.info("👈 Set parameters and click 'Generate Im(s) vs z Analysis' to create a visualization.")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Add footer with instructions
st.markdown("""
---
### Instructions for Using the Dashboard

1. **Select a tab** at the top to choose between Eigenvalue Analysis and Im(s) vs z Analysis
2. **Adjust parameters** in the left panel to configure your analysis
3. **Click the Generate button** to run the analysis with the selected parameters
4. **Explore the results** in the interactive plot
5. For advanced users, you can enable **Debug Mode** to see detailed output

If you encounter any issues with compilation, try clicking the "Recompile C++ Code" button in the sidebar.
""")