import re
import os
import glob

def extract_healpixel(filename):
    """Extract the HEALPix pixel number from a filename like
    'lightcurve_hpx8501.parquet' -> 8501."""
    match = re.search(r'hpx(\d+)', filename)
    if match is None:
        raise ValueError(f"Could not extract healpixel number from filename: {filename!r} "
                         f"(expected a pattern like 'hpx<digits>' somewhere in the name)")
    return int(match.group(1))

def get_filepaths(dirpath, fn):
    filepaths = glob.glob(os.path.join(dirpath, fn))
    if len(filepaths) == 0:
        raise FileNotFoundError(f"No files matching '{fn}' found in {dirpath} "
                            f"-- check if the path is correct and the directory exists")
    return filepaths