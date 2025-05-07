import streamlit as st
import subprocess
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import io

# Set page config
st.set_page_config(
    page_title="Eigenvalue Analysis",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("Eigenvalue Analysis Visualization")
st.markdown("""
This application visualizes eigenvalue analysis for matrices with specific properties.
Adjust the parameters below to generate a plot showing the relationship between empirical
and theoretical eigenvalues.
""")

# Create output directory in the current working directory
current_dir = os.getcwd()
output_dir = os.path.join(current_dir, "output")
os.makedirs(output_dir, exist_ok=True)

# Compile the C++ code at runtime
cpp_file = os.path.join(current_dir, "app.cpp")
executable = os.path.join(current_dir, "eigen_analysis")

# Check if cpp file exists
if not os.path.exists(cpp_file):
    st.error(f"C++ source file not found at: {cpp_file}")
    st.stop()

# Compile the C++ code with the right OpenCV libraries
try:
    st.info("Compiling C++ code...")
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
    
except Exception as e:
    st.error(f"Error during compilation: {str(e)}")
    st.stop()

# Input parameters sidebar
st.sidebar.header("Parameters")

# Parameter inputs with defaults and validation
n = st.sidebar.number_input("Sample size (n)", min_value=5, max_value=100000, value=100, step=5, help="Number of samples")
p = st.sidebar.number_input("Dimension (p)", min_value=5, max_value=1000000, value=50, step=5, help="Dimensionality")
a = st.sidebar.number_input("Value for a", min_value=1.1, max_value=10.0, value=2.0, step=0.1, help="Parameter a > 1")

# Automatically calculate y = p/n (as requested)
y = p/n
st.sidebar.text(f"Value for y = p/n: {y:.4f}")

# Add fineness control
st.sidebar.subheader("Calculation Controls")
fineness = st.sidebar.slider(
    "Beta points", 
    min_value=20, 
    max_value=500, 
    value=100, 
    step=10,
    help="Number of points to calculate along the β axis (0 to 1)"
)

# Add controls for theoretical calculation precision
theory_grid_points = st.sidebar.slider(
    "Theoretical grid points", 
    min_value=100, 
    max_value=1000, 
    value=200, 
    step=50,
    help="Number of points in initial grid search for theoretical calculations"
)

theory_tolerance = st.sidebar.number_input(
    "Theoretical tolerance", 
    min_value=1e-12, 
    max_value=1e-6, 
    value=1e-10, 
    format="%.1e",
    help="Convergence tolerance for golden section search"
)

# Generate button
if st.sidebar.button("Generate Plot", type="primary"):
    # Show progress
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
            status_text.text("Calculations complete! Generating plot...")
            
            # Load the results from the JSON file
            with open(data_file, 'r') as f:
                data = json.load(f)
            
            # Create a better plot with matplotlib
            beta_values = np.array(data['beta_values'])
            max_eigenvalues = np.array(data['max_eigenvalues'])
            min_eigenvalues = np.array(data['min_eigenvalues'])
            theoretical_max = np.array(data['theoretical_max'])
            theoretical_min = np.array(data['theoretical_min'])
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 9), dpi=100)
            
            # Set the background color
            fig.patch.set_facecolor('#f5f5f5')
            ax.set_facecolor('#f0f0f0')
            
            # Plot the data
            ax.plot(beta_values, max_eigenvalues, 'r-', linewidth=2, 
                    label='Empirical Max Eigenvalue', marker='o', markevery=len(beta_values)//20)
            ax.plot(beta_values, min_eigenvalues, 'b-', linewidth=2, 
                    label='Empirical Min Eigenvalue', marker='o', markevery=len(beta_values)//20)
            ax.plot(beta_values, theoretical_max, 'g-', linewidth=2, 
                    label='Theoretical Max Function', marker='D', markevery=len(beta_values)//20)
            ax.plot(beta_values, theoretical_min, 'm-', linewidth=2, 
                    label='Theoretical Min Function', marker='D', markevery=len(beta_values)//20)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Set labels and title
            ax.set_xlabel('β', fontsize=14)
            ax.set_ylabel('Eigenvalues', fontsize=14)
            ax.set_title(f'Eigenvalue Analysis: n={n}, p={p}, a={a}, y={y:.4f}', fontsize=16)
            
            # Add legend
            ax.legend(loc='best', fontsize=12, framealpha=0.9)
            
            # Add formulas as text
            formula_text1 = r"Max Function: $\max_{k \in (0,\infty)} \frac{y\beta(a-1)k + (ak+1)((y-1)k-1)}{(ak+1)(k^2+k)}$"
            formula_text2 = r"Min Function: $\min_{t \in (-1/a,0)} \frac{y\beta(a-1)t + (at+1)((y-1)t-1)}{(at+1)(t^2+t)}$"
            
            plt.figtext(0.02, 0.02, formula_text1, fontsize=10, color='green')
            plt.figtext(0.55, 0.02, formula_text2, fontsize=10, color='purple')
            
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
            
            # Display the image in Streamlit
            status_text.success("Analysis completed successfully!")
            st.image(buf, use_column_width=True)
            
            # Provide download button
            with open(output_file, "rb") as file:
                btn = st.download_button(
                    label="Download Plot",
                    data=file,
                    file_name=f"eigenvalue_analysis_n{n}_p{p}_a{a}_y{y:.4f}.png",
                    mime="image/png"
                )
            
            # Add some statistics
            st.subheader("Statistical Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### Maximum Eigenvalues")
                st.write(f"Empirical Max: {max(max_eigenvalues):.6f}")
                st.write(f"Theoretical Max: {max(theoretical_max):.6f}")
                st.write(f"Difference: {abs(max(max_eigenvalues) - max(theoretical_max)):.6f}")
            
            with col2:
                st.write("### Minimum Eigenvalues")
                st.write(f"Empirical Min: {min(min_eigenvalues):.6f}")
                st.write(f"Theoretical Min: {min(theoretical_min):.6f}")
                st.write(f"Difference: {abs(min(min_eigenvalues) - min(theoretical_min)):.6f}")
            
            # Display calculation settings
            with st.expander("Calculation Settings"):
                st.write(f"Beta points: {fineness}")
                st.write(f"Theoretical grid points: {theory_grid_points}")
                st.write(f"Theoretical tolerance: {theory_tolerance:.1e}")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Show example plot on startup or previous results
example_file = os.path.join(output_dir, "eigenvalue_analysis.png")
if os.path.exists(example_file):
    # Show the most recent plot by default
    st.subheader("Current Plot")
    img = Image.open(example_file)
    st.image(img, use_column_width=True)
else:
    st.info("👈 Set parameters and click 'Generate Plot' to create a visualization.")

# Add information about the analysis
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