import streamlit as st
import subprocess
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import io

# Set page config with wider layout
st.set_page_config(
    page_title="Eigenvalue Analysis Dashboard",
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
    .stats-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stats-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .stats-label {
        font-size: 0.9rem;
        color: #616161;
        margin-top: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# Dashboard Header
st.markdown('<h1 class="main-header">Eigenvalue Analysis Dashboard</h1>', unsafe_allow_html=True)

# Create output directory in the current working directory
current_dir = os.getcwd()
output_dir = os.path.join(current_dir, "output")
os.makedirs(output_dir, exist_ok=True)

# Compile the C++ code at runtime
cpp_file = os.path.join(current_dir, "app.cpp")
executable = os.path.join(current_dir, "eigen_analysis")

# Two-column layout for the dashboard
left_column, right_column = st.columns([1, 3])

with left_column:
    st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
    st.markdown('<div class="panel-header">Control Panel</div>', unsafe_allow_html=True)
    
    # Check if cpp file exists and compile if necessary
    if not os.path.exists(cpp_file):
        st.error(f"C++ source file not found at: {cpp_file}")
        st.stop()
    
    # Compile the C++ code with the right OpenCV libraries
    if not os.path.exists(executable) or st.button("Recompile C++ Code"):
        with st.spinner("Compiling C++ code..."):
            compile_commands = [
                f"g++ -o {executable} {cpp_file} `pkg-config --cflags --libs opencv4` -std=c++11",
                f"g++ -o {executable} {cpp_file} `pkg-config --cflags --libs opencv` -std=c++11",
                f"g++ -o {executable} {cpp_file} -I/usr/include/opencv4 -lopencv_core -lopencv_imgproc -std=c++11"
            ]
            
            compiled = False
            for cmd in compile_commands:
                compile_result = subprocess.run(
                    cmd, 
                    shell=True,
                    capture_output=True,
                    text=True
                )
                
                if compile_result.returncode == 0:
                    compiled = True
                    break
            
            if not compiled:
                st.error("All compilation attempts failed. Please check the system requirements.")
                st.stop()
            
            # Make sure the executable is executable
            os.chmod(executable, 0o755)
            st.success("C++ code compiled successfully")
    
    # Parameter inputs with defaults and validation
    st.markdown("### Matrix Parameters")
    n = st.number_input("Sample size (n)", min_value=5, max_value=1000, value=100, step=5, help="Number of samples")
    p = st.number_input("Dimension (p)", min_value=5, max_value=1000, value=50, step=5, help="Dimensionality")
    a = st.number_input("Value for a", min_value=1.1, max_value=10.0, value=2.0, step=0.1, help="Parameter a > 1")
    
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
        help="Number of points to calculate along the β axis (0 to 1)"
    )
    
    with st.expander("Advanced Settings"):
        # Add controls for theoretical calculation precision
        theory_grid_points = st.slider(
            "Theoretical grid points", 
            min_value=100, 
            max_value=1000, 
            value=200, 
            step=50,
            help="Number of points in initial grid search for theoretical calculations"
        )
        
        theory_tolerance = st.number_input(
            "Theoretical tolerance", 
            min_value=1e-12, 
            max_value=1e-6, 
            value=1e-10, 
            format="%.1e",
            help="Convergence tolerance for golden section search"
        )
    
    # Generate button
    generate_button = st.button("Generate Analysis", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # About section
    with st.expander("About Eigenvalue Analysis"):
        st.markdown("""
        ## Theory
        
        This application visualizes the relationship between empirical and theoretical eigenvalues for matrices with specific properties.
        
        The analysis examines:
        
        - **Empirical Max/Min Eigenvalues**: The maximum and minimum eigenvalues calculated from the generated matrices
        - **Theoretical Max/Min Functions**: The theoretical bounds derived from mathematical analysis
        
        ### Key Parameters
        
        - **n**: Sample size
        - **p**: Dimension
        - **a**: Value > 1 that affects the distribution of eigenvalues
        - **y**: Value calculated as p/n that affects scaling
        
        ### Calculation Controls
        
        - **Beta points**: Number of points calculated along the β range (0 to 1)
        - **Theoretical grid points**: Number of points in initial grid search for finding theoretical max/min
        - **Theoretical tolerance**: Convergence tolerance for golden section search algorithm
        
        ### Mathematical Formulas
        
        Max Function: 
        max{k ∈ (0,∞)} [yβ(a-1)k + (ak+1)((y-1)k-1)]/[(ak+1)(k²+k)]
        
        Min Function: 
        min{t ∈ (-1/a,0)} [yβ(a-1)t + (at+1)((y-1)t-1)]/[(at+1)(t²+t)]
        """)

with right_column:
    # Main visualization area
    st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
    st.markdown('<div class="panel-header">Eigenvalue Analysis Visualization</div>', unsafe_allow_html=True)
    
    # Container for the analysis results
    results_container = st.container()
    
    # Process when generate button is clicked
    if generate_button:
        with results_container:
            # Show progress
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            try:
                # Run the C++ executable with the parameters in JSON output mode
                data_file = os.path.join(output_dir, "eigenvalue_data.json")
                
                # Delete previous output if exists
                if os.path.exists(data_file):
                    os.remove(data_file)
                
                # Execute the C++ program
                cmd = [
                    executable, 
                    str(n), 
                    str(p), 
                    str(a), 
                    str(y), 
                    str(fineness), 
                    str(theory_grid_points),
                    str(theory_tolerance),
                    data_file
                ]
                
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Show output in a status area
                status_text.text("Starting calculations...")
                
                last_progress = 0
                while process.poll() is None:
                    output = process.stdout.readline()
                    if output:
                        if output.startswith("PROGRESS:"):
                            try:
                                # Update progress bar
                                progress_value = float(output.split(":")[1].strip())
                                progress_bar.progress(progress_value)
                                last_progress = progress_value
                                status_text.text(f"Calculating... {int(progress_value * 100)}% complete")
                            except:
                                pass
                        else:
                            status_text.text(output.strip())
                    time.sleep(0.1)
                
                return_code = process.poll()
                
                if return_code != 0:
                    error = process.stderr.read()
                    st.error(f"Error executing the analysis: {error}")
                else:
                    progress_bar.progress(1.0)
                    status_text.text("Calculations complete! Generating visualization...")
                    
                    # Load the results from the JSON file
                    with open(data_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract data
                    beta_values = np.array(data['beta_values'])
                    max_eigenvalues = np.array(data['max_eigenvalues'])
                    min_eigenvalues = np.array(data['min_eigenvalues'])
                    theoretical_max = np.array(data['theoretical_max'])
                    theoretical_min = np.array(data['theoretical_min'])
                    
                    # Create the plot
                    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
                    
                    # Set the background color
                    fig.patch.set_facecolor('#f9f9f9')
                    ax.set_facecolor('#f0f0f0')
                    
                    # Plot the data with improved styling
                    ax.plot(beta_values, max_eigenvalues, 'r-', linewidth=2.5, 
                            label='Empirical Max Eigenvalue', marker='o', markevery=len(beta_values)//20, markersize=6)
                    ax.plot(beta_values, min_eigenvalues, 'b-', linewidth=2.5, 
                            label='Empirical Min Eigenvalue', marker='o', markevery=len(beta_values)//20, markersize=6)
                    ax.plot(beta_values, theoretical_max, 'g-', linewidth=2.5, 
                            label='Theoretical Max Function', marker='D', markevery=len(beta_values)//20, markersize=6)
                    ax.plot(beta_values, theoretical_min, 'm-', linewidth=2.5, 
                            label='Theoretical Min Function', marker='D', markevery=len(beta_values)//20, markersize=6)
                    
                    # Add grid
                    ax.grid(True, linestyle='--', alpha=0.7)
                    
                    # Set labels and title with better formatting
                    ax.set_xlabel('β Parameter', fontsize=14, fontweight='bold')
                    ax.set_ylabel('Eigenvalues', fontsize=14, fontweight='bold')
                    ax.set_title(f'Eigenvalue Analysis: n={n}, p={p}, a={a}, y={y:.4f}', 
                                 fontsize=16, fontweight='bold', pad=15)
                    
                    # Add legend with improved styling
                    legend = ax.legend(loc='best', fontsize=12, framealpha=0.9, 
                                      fancybox=True, shadow=True, borderpad=1)
                    
                    # Add formulas as text with better styling
                    formula_text1 = r"Max Function: $\max_{k \in (0,\infty)} \frac{y\beta(a-1)k + (ak+1)((y-1)k-1)}{(ak+1)(k^2+k)}$"
                    formula_text2 = r"Min Function: $\min_{t \in (-1/a,0)} \frac{y\beta(a-1)t + (at+1)((y-1)t-1)}{(at+1)(t^2+t)}$"
                    
                    plt.figtext(0.02, 0.02, formula_text1, fontsize=10, color='green', 
                               bbox=dict(facecolor='white', alpha=0.8, edgecolor='green', boxstyle='round,pad=0.5'))
                    plt.figtext(0.55, 0.02, formula_text2, fontsize=10, color='purple',
                               bbox=dict(facecolor='white', alpha=0.8, edgecolor='purple', boxstyle='round,pad=0.5'))
                    
                    # Adjust layout
                    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
                    
                    # Save the plot to a buffer
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=100)
                    buf.seek(0)
                    
                    # Save to file
                    output_file = os.path.join(output_dir, "eigenvalue_analysis.png")
                    plt.savefig(output_file, format='png', dpi=100)
                    plt.close()
                    
                    # Clear progress container
                    progress_container.empty()
                    
                    # Display the image in Streamlit (with fixed deprecated parameter)
                    st.image(buf, use_container_width=True)
                    
                    # Provide download button
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        with open(output_file, "rb") as file:
                            btn = st.download_button(
                                label="Download Plot",
                                data=file,
                                file_name=f"eigenvalue_analysis_n{n}_p{p}_a{a}_y{y:.4f}.png",
                                mime="image/png",
                                use_container_width=True
                            )
                    
                    # Add statistics section with cards
                    st.markdown("### Results Summary")
                    
                    # Calculate key statistics
                    emp_max = max(max_eigenvalues)
                    emp_min = min(min_eigenvalues)
                    theo_max = max(theoretical_max)
                    theo_min = min(theoretical_min)
                    max_diff = abs(emp_max - theo_max)
                    min_diff = abs(emp_min - theo_min)
                    
                    # Display statistics in a card layout
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown('<div class="stats-card">', unsafe_allow_html=True)
                        st.markdown(f'<div class="stats-value">{emp_max:.4f}</div>', unsafe_allow_html=True)
                        st.markdown('<div class="stats-label">Empirical Maximum</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="stats-card">', unsafe_allow_html=True)
                        st.markdown(f'<div class="stats-value">{emp_min:.4f}</div>', unsafe_allow_html=True)
                        st.markdown('<div class="stats-label">Empirical Minimum</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown('<div class="stats-card">', unsafe_allow_html=True)
                        st.markdown(f'<div class="stats-value">{theo_max:.4f}</div>', unsafe_allow_html=True)
                        st.markdown('<div class="stats-label">Theoretical Maximum</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown('<div class="stats-card">', unsafe_allow_html=True)
                        st.markdown(f'<div class="stats-value">{theo_min:.4f}</div>', unsafe_allow_html=True)
                        st.markdown('<div class="stats-label">Theoretical Minimum</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown('<div class="stats-card">', unsafe_allow_html=True)
                        st.markdown(f'<div class="stats-value">{max_diff:.4f}</div>', unsafe_allow_html=True)
                        st.markdown('<div class="stats-label">Max Difference</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="stats-card">', unsafe_allow_html=True)
                        st.markdown(f'<div class="stats-value">{min_diff:.4f}</div>', unsafe_allow_html=True)
                        st.markdown('<div class="stats-label">Min Difference</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Add calculation settings
                    with st.expander("Calculation Details"):
                        st.markdown(f"""
                        - **Matrix Dimensions**: {n} × {p}
                        - **Parameter a**: {a}
                        - **Parameter y (p/n)**: {y:.4f}
                        - **Beta points**: {fineness}
                        - **Theoretical grid points**: {theory_grid_points}
                        - **Theoretical tolerance**: {theory_tolerance:.1e}
                        """)
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    else:
        # Check for existing results
        example_file = os.path.join(output_dir, "eigenvalue_analysis.png")
        if os.path.exists(example_file):
            # Show the most recent plot by default
            st.image(example_file, use_container_width=True)
            st.info("This is the most recent analysis result. Adjust parameters and click 'Generate Analysis' to create a new visualization.")
        else:
            # Show placeholder
            st.info("👈 Set parameters and click 'Generate Analysis' to create a visualization.")
    
    st.markdown('</div>', unsafe_allow_html=True)