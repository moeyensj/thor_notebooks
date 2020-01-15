RUN_DIR = "/Users/moeyensj/Google Drive/Astronomy/Data/thor/results/ztf/run_16/"
THOR_DIR = "/Users/moeyensj/projects/thor/thor"

import os
import sys
import time
import pandas as pd
import numpy as np
import multiprocessing as mp
from functools import partial

sys.path.append(THOR_DIR)

from thor.config import Config
from thor.constants import Constants as c
from thor.orbits.iod import iod

MU = c.G * c.M_SUN
THREADS = 28

def _iod(obs_ids,
         cluster_id,
         observations=None,
         observation_selection_method="combinations",
         iterate=True, 
         light_time=True,
         max_iter=50, 
         tol=1e-15, 
         propagatorKwargs={
            "observatoryCode" : "I11",
            "mjdScale" : "UTC",
            "dynamical_model" : "2",
         },
         mu=MU, 
         columnMapping=Config.columnMapping):

    orbit, best_obs_ids, min_chi2 = iod(
        observations[observations[columnMapping["obs_id"]].isin(obs_ids)],
        observation_selection_method=observation_selection_method,
        iterate=iterate, 
        light_time=light_time,
        max_iter=max_iter, 
        tol=tol, 
        propagatorKwargs=propagatorKwargs,
        mu=mu, 
        columnMapping=columnMapping)
    
    if orbit is not None:
        orbit_id = "{}".format("_".join(best_obs_ids.astype(str)))
        return [orbit_id, cluster_id, *orbit[:], min_chi2]
    else:
        return []

def initialOrbitDetermination(observations, 
                              clusterMembers, 
                              observation_selection_method="combinations", 
                              iterate=True, 
                              light_time=True, 
                              max_iter=50,
                              tol=1e-15, 
                              propagatorKwargs={
                                "observatoryCode" : "I41",
                                "mjdScale" : "UTC",
                                "dynamical_model" : "2",
                               },
                              mu=MU, 
                              threads=10,
                              columnMapping=Config.columnMapping):

    grouped = clusterMembers.groupby(by="cluster_id")[columnMapping["obs_id"]].apply(list)
    cluster_ids = grouped.index.values
    obs_ids = grouped.values

    orbits = []
    if threads > 1:
        p = mp.Pool(threads)
        orbits = p.starmap(partial(_iod,
                                   observations=observations,
                                   observation_selection_method=observation_selection_method,
                                   iterate=iterate, 
                                   light_time=light_time,
                                   max_iter=max_iter, 
                                   tol=tol, 
                                   propagatorKwargs=propagatorKwargs,
                                   mu=mu, 
                                   columnMapping=columnMapping),
                           zip(obs_ids, cluster_ids))

    orbits_df = pd.DataFrame(orbits, columns=["orbit_id", "cluster_id", "epoch_mjd", "obj_x", "obj_y", "obj_z", "obj_vx", "obj_vy", "obj_vz", "chi2"])
    
    # Remove empty rows
    orbits_df.dropna(inplace=True)
    return orbits_df
                                   

columnMapping = {        
        
        # Observation ID
        "obs_id" : "obs_id",
        
        # Exposure time
        "exp_mjd" : "exp_mjd",
        
        # Visit ID
        "visit_id" : "visit_id",
        
        # Field ID
        "field_id" : "field",
        
        # Field RA in degrees
        "field_RA_deg" : "fieldRA_deg",
        
        # Field Dec in degrees
        "field_Dec_deg" : "fieldDec_deg",
        
        # Night number
        "night": "nid",
        
        # RA in degrees
        "RA_deg" : "ra",
        
        # Dec in degrees
        "Dec_deg" : "decl",
        
        # Observer's x coordinate in AU
        "obs_x_au" : "HEclObsy_X_au",
        
        # Observer's y coordinate in AU
        "obs_y_au" : "HEclObsy_Y_au",
        
        # Observer's z coordinate in AU
        "obs_z_au" : "HEclObsy_Z_au",
        
        # Magnitude (UNUSED)
        "mag" : "magpsf",
        
        ### Truth Parameters
        
        # Object name
        "name" : "designation",
        
        # Observer-object distance in AU
        "Delta_au" : "Delta_au",
        
        # Sun-object distance in AU (heliocentric distance)
        "r_au" : "r_au",
        
        # Object's x coordinate in AU
        "obj_x_au" : "HEclObj_X_au",
        
        # Object's y coordinate in AU
        "obj_y_au" : "HEclObj_Y_au",
        
        # Object's z coordinate in AU
        "obj_z_au" : "HEclObj_Z_au",
        
        # Object's x velocity in AU per day
        "obj_dx/dt_au_p_day" : "HEclObj_dX/dt_au_p_day",
        
        # Object's y velocity in AU per day
        "obj_dy/dt_au_p_day" : "HEclObj_dY/dt_au_p_day",
        
        # Object's z velocity in AU per day
        "obj_dz/dt_au_p_day" : "HEclObj_dZ/dt_au_p_day",
        
        # Semi-major axis
        "a_au" : "a_au",
        
        # Inclination
        "i_deg" : "i_deg",
        
        # Eccentricity
        "e" : "e",
    }

IOD_DIR = os.path.join(RUN_DIR, "iod_orbits")
if not os.path.exists(IOD_DIR):
    os.mkdir(IOD_DIR)
test_orbits = pd.read_csv(os.path.join(RUN_DIR, "orbits.txt"), sep=" ", index_col=False)

time_start = time.time()
for orbit_id in test_orbits["orbit_id"].unique()[407:]:

    print("THOR: Initial Orbit Determination")
    print("-------------------------")
    print("Running orbit {}...".format(orbit_id))
    print("")

    orbit_dir = os.path.join(RUN_DIR, "orbit_{:04d}".format(orbit_id))
    
    allClusters = pd.read_csv(os.path.join(orbit_dir, "allClusters.txt"), sep=" ", index_col=False)
    clusterMembers = pd.read_csv(os.path.join(orbit_dir, "clusterMembers.txt"), sep=" ", index_col=False)
    projected_obs = pd.read_csv(os.path.join(orbit_dir, "projected_obs.txt"), sep=" ", index_col=False)
    
    astrometric_err = 1/3600/10 # in degrees
    #projected_obs["RA_deg"] = projected_obs["RA_deg"] + np.random.normal(loc=0, scale=astrometric_err, size=len(projected_obs)) * np.cos(np.radians(projected_obs["Dec_deg"].values))
    #projected_obs["Dec_deg"] = projected_obs["Dec_deg"] + np.random.normal(loc=0, scale=astrometric_err, size=len(projected_obs))

    columnMapping["RA_sigma_deg"] = "RA_sigma_deg"
    columnMapping["Dec_sigma_deg"] = "Dec_sigma_deg"
    projected_obs["RA_sigma_deg"] = astrometric_err * np.ones(len(projected_obs))
    projected_obs["Dec_sigma_deg"] = astrometric_err * np.ones(len(projected_obs))

    print("Number of clusters: {}".format(clusterMembers["cluster_id"].nunique()))
    
    iod_orbits = initialOrbitDetermination(projected_obs, 
                              clusterMembers, 
                              observation_selection_method="combinations", 
                              iterate=True, 
                              light_time=True, 
                              max_iter=50,
                              tol=1e-15, 
                              propagatorKwargs={
                                "observatoryCode" : "I41",
                                "mjdScale" : "UTC",
                                "dynamical_model" : "2",
                               },
                              mu=MU, 
                              threads=THREADS,
                              columnMapping=columnMapping)
    
    iod_orbits = iod_orbits.merge(allClusters[["cluster_id", "linked_object", "pure", "partial", "false"]], on="cluster_id")
    iod_orbits["orbit_id"] = [orbit_id for i in range(len(iod_orbits))]
    iod_orbits["linked_object"] = iod_orbits["linked_object"].astype(str)
    iod_orbits["cluster_id"] = iod_orbits["cluster_id"].astype(int)
    
    iod_orbits.to_csv(os.path.join(IOD_DIR, "iod_orbit_{:04d}.txt".format(orbit_id)), sep=" ", index=False)
    
    print("Found {} preliminary orbits.".format(len(iod_orbits)))
    print("")

print("Total time: {}".format(time.time() - time_start))
    