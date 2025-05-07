import streamlit as st
import subprocess
import os
from PIL import Image
import time

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

# Create output directory if it doesn't exist
os.makedirs("/app/output", exist_ok=True)

# Input parameters sidebar
st.sidebar.header("Parameters")

# Parameter inputs with defaults and validation
col1, col2 = st.sidebar.columns(2)
with col1:
    n = st.number_input("Sample size (n)", min_value=5, max_value=1000, value=100, step=5, help="Number of samples")
    a = st.number_input("Value for a", min_value=1.1, max_value=10.0, value=2.0, step=0.1, help="Parameter a > 1")

with col2:
    p = st.number_input("Dimension (p)", min_value=5, max_value=1000, value=50, step=5, help="Dimensionality")
    y = st.number_input("Value for y", min_value=0.1, max_value=10.0, value=1.0, step=0.1, help="Parameter y > 0")

# Generate button
if st.sidebar.button("Generate Plot", type="primary"):
    # Show progress
    with st.spinner("Generating eigenvalue analysis plot... This may take a few moments."):
        # Run the C++ executable with the parameters
        output_file = "/app/output/eigenvalue_analysis.png"
        
        # Delete previous output if exists
        if os.path.exists(output_file):
            os.remove(output_file)
        
        # Execute the C++ program
        try:
            cmd = ["/app/eigen_analysis", str(n), str(p), str(a), str(y)]
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Show output in a status area
            status_area = st.empty()
            
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    status_area.info(output.strip())
            
            return_code = process.poll()
            
            if return_code != 0:
                error = process.stderr.read()
                st.error(f"Error executing the analysis: {error}")
            else:
                status_area.success("Analysis completed successfully!")
                
                # Wait a moment to ensure the file is written
                time.sleep(1)
                
                # Display the image if it exists
                if os.path.exists(output_file):
                    img = Image.open(output_file)
                    st.image(img, use_column_width=True)
                    
                    # Provide download button
                    with open(output_file, "rb") as file:
                        btn = st.download_button(
                            label="Download Plot",
                            data=file,
                            file_name=f"eigenvalue_analysis_n{n}_p{p}_a{a}_y{y}.png",
                            mime="image/png"
                        )
                else:
                    st.error("Plot generation failed. Output file not found.")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Show example plot on startup
if not os.path.exists("/app/output/eigenvalue_analysis.png"):
    st.info("👈 Set parameters and click 'Generate Plot' to create a visualization.")
else:
    # Show the most recent plot by default
    st.subheader("Current Plot")
    img = Image.open("/app/output/eigenvalue_analysis.png")
    st.image(img, use_column_width=True)

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
    - **y**: Value that affects scaling
    
    ### Mathematical Formulas
    
    Max Function: 
    max{k ∈ (0,∞)} [yβ(a-1)k + (ak+1)((y-1)k-1)]/[(ak+1)(k²+k)y]
    
    Min Function: 
    min{t ∈ (-1/a,0)} [yβ(a-1)t + (at+1)((y-1)t-1)]/[(at+1)(t²+t)y]
    """)