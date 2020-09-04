import pandas as pd

def fitres_to_pd(filename, verbose=False):
    """Takes in a FITRES file and outputs a Pandas dataframe.

    Reads a FITRES file with a single table named SN and outputs the columns to
    a Pandas DataFrame. Columns are converted to numeric types where possible.
    If verbose mode is enabled, warnings are printed when this fails.

    Parameters:
        filename (str): The file
        verbose (bool): Flag to indicate warnings should be printed

    Returns:
        pandas.DataFrame: A dataframe containing the FITRES table
    """
    data = []

    with open(filename) as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if line.startswith("VARNAMES:"):
                line = line[len("VARNAMES:") :]
                line = line.replace(",", " ")
                line = line.replace("\n", "")
                variables = line.split()

            elif line.startswith("SN:"):
                line = line[len("SN:") :]
                line = line.split()
                data.append(line)

            elif line.startswith("GAL:"):
                line = line[len("GAL:") :]
                line = line.split()
                data.append(line)

    df = pd.DataFrame(data, columns=variables)

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError as e:
            if verbose:
                print(
                    "While converting "
                    + filename
                    + " to dataframe (columns to numeric): "
                    + str(e)
                )
            continue
    return df


def calculate_vpec(zobs, ztrue, eobs=None, etrue=None):
    """Calculates the peculiar velocity of an object in km/s.

    Takes the observed and true redshifts of an object and calculates
    the peculiar velocity using the formula (1 + zobs) = (1 + ztrue)*(1 + zpec).

    Parameters:
        zobs (float): The observed redshift
        ztrue (float): The true (cosmological) redshift
        eobs (float): The error in zobs (optional)
        etrue (float): The error in ztrue (optional)
    Returns:
        float: The peculiar velocity in km/s
        float: The error in peculiar velocity in km/s
    """
    import numpy as np

    c = 299792.458  # km/s

    vpec = c * ((1.0 + zobs) / (1.0 + ztrue) - 1.0)

    if eobs is None and etrue is None:
        return vpec
    elif eobs is None or etrue is None:
        raise ValueError("Need to specify either both or neither error.")
    else:
        error = (vpec + c) * np.sqrt(
            np.power(eobs / (zobs + 1.0), 2) + np.power(etrue / (ztrue + 1.0), 2)
        )
        return (vpec, error)


def mu_to_z(mus, emus=None, cosmo=None):
    """Converts distance modulus to redshift.

    Takes a list/scalar of distance moduli and (optionally) errors
    and transforms them corresponding to the Planck cosmology or
    a passed cosmology.

    Parameters:
        mus (float): The distance moduli
        emus (float): The errors in distance modulus
        cosmo (astropy.cosmology.FLRW): The cosmology
    Returns:
        zs (float): The redshifts
        emus (float): The errors in redshift)
    """
    import astropy.units as u
    from astropy.cosmology import z_at_value
    import numpy as np

    if cosmo is None:
        from astropy.cosmology import Planck as cosmo

    if hasattr(mus, "__len__"):
        if emus is not None:
            # Check our lists are the same length
            assert len(mus) == len(emus)

            zs = []
            ezs = []

            for mu, emu in zip(mus, emus):
                zs.append(z_at_value(cosmo.distmod, mu * u.mag))
                ezs.append(
                    np.abs(
                        z_at_value(cosmo.distmod, (mu + emu) * u.mag)
                        - z_at_value(cosmo.distmod, mu * u.mag)
                    )
                )
            return (np.array(zs), np.array(ezs))
        else:
            zs = []
            for mu in mus:
                zs.append(z_at_value(cosmo.distmod, mu * u.mag))
            return np.array(zs)
    else:
        mu = mus
        emu = emus

        if emus is not None:
            return (
                z_at_value(cosmo.distmod, mu * u.mag),
                np.abs(
                    z_at_value(cosmo.distmod, (mu + emu) * u.mag)
                    - z_at_value(cosmo.distmod, mu * u.mag)
                ),
            )
        else:
            return z_at_value(cosmo.distmod, mu * u.mag)
