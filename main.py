import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

st.title("Plasmon Catalysis Data Fitting")
st.markdown(
    r"""
    ## Theory

    This app fits linear and Arrhenius models to provided data for plasmon catalysis experiments.
    The data provided must show the dependance of reaction rate $k$ on illumination intensity $I$.  

    Briefly, for reactions explained by hot-electron mechanism, $k$ is expected to increase linearly increasing illumination intensity.
    For thermal reactions, $k$ is expected to follow the Arrhenius equation as nanoparticles are heated by illumination. See 10.1038/s41377-020-00345-0 for more details.  
    
    The linear model is given by:
    $$
    k = a I + b
    $$

    The Arrhenius model is given by: 
    $$
    y = A \exp\Big(-\frac{E_a}{R(T_0 + c I)}\Big)
    $$
    where T_0 + c I is the temperature of the nanoparticle surface, given by ambient temperature $T_0$ and a local NP heating. 
    Local heating is proportional to the illumination intensity $I$ with a constant $c$. Value of the constant $c$ 
    will be fitted to the data and can be used to validate if Arrhenius model is appropriate for the data. For small gold nanoparticles (smaller than 100 nm)
    in water, $c$ is on the order of $5 r_{np}$ (in units of $[K m^2 / W]$) where $r_{np}$ is the radius of the nanoparticle.
    Note that the illumination intensity $I$ must be in SI units of $[W/m^2]$ if you want to compare directly fitted $c$ to the expected value of $5 r_{np}$.
    
    ## Using the App

    Data can be uploaded as a CSV or Excel file or entered manually. 
    For uploaded data, the first column is assumed to be the intensity $I$ and the second column is the rate $k$.
    For manual entry, the first column is $I$ and the second column is $k$.
    The app will fit the data to the models and plot the results.
    """
)


# Example fitting model (modify as needed)
def arrhenius(x, a, e, c):
    R = 8.314
    T = T_0 + x * c
    return a * np.exp(-e / (R * T))


def model(x, a, e, c):
    return np.log(arrhenius(x, a, e, c))

def linear_model(x, a, b):
    return a + b * x



# Option 1: File Upload
st.subheader("Upload CSV File (X, Y)")
uploaded_file = st.file_uploader("Choose a table file", type=["csv", "xsl", "xlsx"])

if uploaded_file:
    for reader in [pd.read_csv, pd.read_excel]:
        try:
            df = reader(uploaded_file)
            break
        except:
            pass
    else:
        raise ValueError("Invalid file format")
    st.write("Uploaded Data:")
else:
    # Option 2: Manual Data Entry (Spreadsheet)
    st.subheader("Or Enter Data Manually")
    df = pd.DataFrame({"I": np.arange(1, 6), "k": np.exp(-20/np.arange(1, 6))})
    df = st.data_editor(df, num_rows="dynamic")

x_data = df.values[:, 0]
y_data = df.values[:, 1]

# Inputs for initial guess of model parameters
st.subheader("Initial Guess for Model Parameters")
a_guess = st.number_input("A", value=y_data.mean())
e_guess = st.number_input("E_a", value=1000.0)
c_guess = st.number_input("C", value=0.0001)
T_0 = st.number_input("Ambient temperature assumed", value=298.0)

# Ensure data is not empty
if not df.empty:
    st.write("Final Data Used for Fitting:")
    st.write(df)

    # Convert DataFrame to numpy for fitting

    if st.button("Run Fitting"):
        try:
            for method in ["lm", "trf", "dogbox"]:
                try:
                    params, _ = curve_fit(model, x_data, np.log(y_data), p0=[a_guess, e_guess, c_guess], method=method)
                    break
                except:
                    pass
            else:
                raise ValueError("Fitting failed")
            a_fit, b_fit, c_fit = params
            st.success(f"Arrhenius parameters: A={a_fit:.3e}, E_a={b_fit:.3e}, C={c_fit:.3e}")

            # Plot data and fit
            # Calculate R^2 for Arrhenius model
            residuals = y_data - arrhenius(x_data, *params)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_data - np.mean(y_data))**2)
            r2_arrhenius = 1 - (ss_res / ss_tot)

            fig, ax = plt.subplots()
            ax.scatter(x_data, y_data, label="Data", color="blue")
            ax.plot(x_data, arrhenius(x_data, *params), label=f"Arrhenius Fit (R²={r2_arrhenius:.3f})", color="red")
            ax.legend()
            ax.set_title(f"A={a_fit:.3e}, E_a={b_fit:.3e}, C={c_fit:.3e}")
            ax.set_xlabel(df.columns[0])
            ax.set_ylabel(df.columns[1])
            st.pyplot(fig)

            # Fit linear model for comparison
            linear_params, _ = curve_fit(linear_model, x_data, y_data, p0=[0, 1])
            a_linear, b_linear = linear_params

            # Calculate R^2 for linear model
            residuals_linear = y_data - linear_model(x_data, *linear_params)
            ss_res_linear = np.sum(residuals_linear**2)
            ss_tot_linear = np.sum((y_data - np.mean(y_data))**2)
            r2_linear = 1 - (ss_res_linear / ss_tot_linear)

            st.success(f"Linear parameters: a={a_linear:.3e}, b={b_linear:.3e}")

            # Plot linear fit
            fig, ax = plt.subplots()
            ax.scatter(x_data, y_data, label="Data", color="blue")
            ax.plot(x_data, linear_model(x_data, *linear_params), label=f"Linear Fit (R²={r2_linear:.3f})", color="green")
            ax.legend()
            ax.set_title(f"a={a_linear:.3e}, b={b_linear:.3e}")
            ax.set_xlabel(df.columns[0])
            ax.set_ylabel(df.columns[1])
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Fitting error: {e}")
