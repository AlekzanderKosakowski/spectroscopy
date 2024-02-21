from scipy.interpolate import LinearNDInterpolator, interp1d
import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits
import pandas as pd
import sys


def smooth(wave, flux, nbin):
    '''
    Transform data into a Pandas DataFrame and use pd.rolling() to easily apply a rolling average to smooth it
    Return array of smoothed values
    '''
    df = pd.DataFrame({"wave":wave, "flux":flux})
    df = df.rolling(nbin, min_periods=1).mean()
    smooth_wave = np.array(df["wave"].values)
    smooth_flux = np.array(df["flux"].values)
    
    return(smooth_wave, smooth_flux)


def mark_feature(lambda_, name=None):
    '''
    Apply vertical lines to the 1D spectrum subplot
    '''
    # h_lines = [6562.79, 4861.35, 4340.472, 4101.734, 3970.075, 3889.064, 3835.397, 3797.9350, 3770.6671, 3750.1883, 3721.946, 3734.369, 8665.02, 8750.46, 8862.89]
    # he_lines = [4026.1914, 4921.9313, 5875.621, 4471.6, 4026.1914, 4471.6,  4921.9313, 5875.621]
    # ca_lines = [5021.14, 3933.66, 3968.47, 4226.74, 8662.170, 8542.144, 8498.062, 5021.14, 8662.170]
    # mg_lines = [4481.327, 5167.327, 5172.698, 5183.619, 4384.637, 5069.802, 6346.962]
    # o_lines = [7775.39, 4701.184, 6156.77, 6158.18, 4046.113, 4701.184]
    # fe_lines = [5169.0282, 5171.4384, 5172.6843, 4920.51, 4383.56, 5196.3391, 4920.51]
    # na_lines = [5889.973, 5895.940, 4249.410, 4249.410]
                
    ax2.axvline(lambda_, linestyle='--', linewidth=0.25, label=name if name else "", alpha=1.0)



def mark_RV(lambda_, amplitude, color='cyan', marker='.', size=5):
    '''
    Add sinusoidally varying markers to the trailed spectrum.
    '''
    for k in np.linspace(0.0, 2.0, 100):
        x0 = lambda_ + amplitude*np.sin(2*np.pi*k - np.pi)
        ax.scatter(x0, k, color=color, s=size, marker=marker, zorder=9, alpha=0.15)



if __name__ =="__main__":
    #
    # Create a trailed spectrum plot with the 1D spectrum below it.
    #
    # TODO: argparse for input files,
    #                    wmin & wmax
    #       Accept text files instead of fits files
    #
    fig = plt.figure(figsize=(12,7))
    ax = fig.add_subplot(5,5,(1,20)) # Trailed spectrum axis
    ax2 = fig.add_subplot(5,5,(21,25)) # Spectrum axis
    plt.subplots_adjust(hspace=0.05)

    path = "/Users/kastra/research/projects/spectroscopy/"

    files = [path+k for k in os.listdir(path) if ".fits" in k and "sum" not in k]
    if len(files) < 5:
        print(f"Too few spectra found ({len(files)}).")
        print("Try to have at least five spectra with good phase coverage.")
        sys.exit()

    # Adjust the lower and upper bounds for the wavelength grid. This will affect the image colorbar scaling.
    wmin = 3600
    wmax = 9500

    min_x = []
    max_x = []
    observed_phase = []

    # Read in data to generate observed grids.
    x, y, z = [np.array([]) for k in range(3)]
    for i,filename in enumerate(sorted(files)):

        file = fits.open(filename)
        w0 = file[0].header["CRVAL1"]
        dw = file[0].header["CD1_1"]

        flux = file[0].data
        while len(flux.shape)>1:
            flux = flux[0]
        wave = np.array([w0 + k*dw for k in range(len(flux))])

        ind = np.where((wave>=wmin) & (wave<=wmax))
        flux = flux[ind]
        wave = wave[ind]

        min_x.append(min(wave))
        max_x.append(max(wave))

        fnu = wave**2 / 299792458e2  * (flux*1e-17)
        fnu /= np.median(fnu)

        wave, fnu = smooth(wave, fnu, 3) # Smooth the data

        try:
            phase = np.ones_like(wave)*file[0].header["phase"] # Orbital phase of each spectrum is assumed to be in the header already. 
            observed_phase.append(file[0].header["phase"])
        except KeyError:
            print(f"Unable to find header KEY='phase' in {filename}\n")
            print(f"This can be added manually with IRAF's 'hedit' or with astropy's 'fits.setval()'\n")
            print(f"    See: https://docs.astropy.org/en/stable/generated/examples/io/modify-fits-header.html")
            sys.exit()

        testing = False
        if testing:
            # Artificially adjust the flux along a row (observed phase value) to confirm the trailed spectrum at "phase+1.0" is identical to at "phase"
            if file[0].header["phase"] > 0.30 and file[0].header["phase"] < 0.40:
                fnu += 2

            # Artificially adjust the flux along a column (wavelength range) to confirm the trailed spectrum
            ind = np.where((wave>4500) & (wave<4510))
            fnu[ind] += 1.5

        x = np.hstack((x, wave))   # Observed Wavelength grid
        y = np.hstack((y, phase))  # Observed Phase grid
        z = np.hstack((z, fnu))    # Observed Flux grid

    # Copy/Paste data for phase 0-1 to phase 1-2
    x = np.hstack((x, x))
    y = np.hstack((y, y+1.0))
    z = np.hstack((z, z))

    # Generate meshgrid for colormap
    X = np.linspace(max(min_x), min(max_x), int(len(x)/len(files)))
    Y = np.linspace(0, 2.0, 200) # Desired phase grid. Higher resolution Y-axis is slower due to more interpolation, but may produce a better-looking image.

    X, Y = np.meshgrid(X, Y)
    Z = LinearNDInterpolator(list(zip(x,y)), z, rescale=True)(X, Y) # Must keep rescale=True or the final plot looks like junk.

    ax.pcolormesh(X, Y, Z, shading='auto', cmap="binary_r", vmin=0.5) # Adjust vmin and vmax to emphasize specific features.

    ax.set_ylim(0.0, 2.0)
    ax.set_ylabel("Orbital Phase")
    ax.tick_params(labelbottom=False,)
    ax2.sharex(ax)

    ax2.set_xlabel(r"Wavelength (${\rm \AA}$)")
    ax2.set_ylabel(r"Flux")


    # Add visual markers to the plot to help locate specific features.
    mark_feature(5889.973)
    mark_RV(5889.973, -20)

    for p in observed_phase: # Mark the locations of the observed orbital phases
        ax.axhline(p, color='black', linewidth=0.5, linestyle='--', label="Observed Spectra")

    # Show the legend, but remove duplicate legend entries from spamming the observed phase horizontal lines.
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())


    # Add the 1D spectrum to ax2
    filename = "./summed_spectrum.fits"
    file = fits.open(filename)
    flux = file[0].data
    while len(flux.shape)>1:
        flux = flux[0]
    w0 = file[0].header["CRVAL1"]
    dw = file[0].header["CD1_1"]
    wave = np.array([w0 + k*dw for k in range(len(flux))])
    fnu = wave**2 / 299792458e2 * flux*1e-17
    wave, fnu = smooth(wave, fnu, 3)
    ax2.plot(wave, fnu/np.median(fnu), linewidth=0.5, color='black')

    plt.show()