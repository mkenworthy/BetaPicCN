rule CN_xcor:
    cache:
        True
    input:
        "src/data/CN/CN_0010K.npy",
        "src/data/CN/CN_0020K.npy",
        "src/data/CN/CN_0050K.npy",
        "src/data/CN/CN_0100K.npy",
        "src/data/CN/CN_0200K.npy",
        "src/data/CN/CN_0300K.npy",
        "src/data/CN/CN_1000K.npy",
        "src/data/CN/CN_2000K.npy",
        "src/data/CN/CN_wavelengths.npy",
        "src/data/2d_nspec.fits",
        "src/data/2d_spec.fits",
        "src/data/2d_wavelength.fits",
        "src/data/2d_baryvel.fits"
    output:
        "src/data/results_HARPS.txt"
    conda:
        "environment.yml"
    script:
        "src/scripts/analyse_HARPS_spectra.py"

rule CCF_single:
    cache:
        False
    input:
    output:
        "src/data/ccf_data1.ecsv"
    conda:
        "environment.yml"
    script:
        "src/scripts/calc_ccf.py"

rule CCF_mean:
    cache:
        False
    input:
    output:
        "src/data/ccf_data1_mean.ecsv"
    conda:
        "environment.yml"
    script:
        "src/scripts/calc_ccf_mean.py"