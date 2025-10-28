"""Erosion and Elevation Tracking Component

Chris Sheehan
"""

import numpy as np
from matplotlib import pyplot as plt
from landlab import Component, imshow_grid


class ErosionElevationTracker(Component):
    
    """Component that tracks, outputs, plots, and maps erosional and elevational 
    changes over time. The component has two modes: “bedrock__and__soil” = True 
    and False. 

    Case: “bedrock__and__soil” = False

        This mode should be used when the grid does not differentiate between bedrock 
        and soil (i.e. when the grid neither contains “bedrock__elevation” nor 
        “soil__depth” fields).

        When the component initiates, it stores a copy of the “topographic__elevation” 
        field (“elevation0”). 
        
        When “run_one_step” is called, the component compares the current “topographic__elevation” 
        field to “elevation0” and calculates the amount of topographic change 
        at each node (“dz”), the average rate of topographic change at each node 
        (“dzdt”), and the mean / standard deviation of topographic change rate 
        across the entire grid (“dzdt_mean” and “dzdt_std”). If an array of total 
        uplift at each node since initiation (“uplift”) is provided to the “run_one_step” 
        method, the component also calculates the amount of erosion at each node 
        (“erosion”), the average erosion rate at each node (“erosionrate”), and 
        the mean / standard deviation of erosion rates across the entire grid 
        (“erosionrate_mean” and “erosionrate_std”). Note that negative erosion 
        values imply deposition. If no “uplift” is provided, the value defaults 
        to 0, meaning that erosional values will be equal to elevational values. 
        After the calculations, the component overwrites “elevation0” with the 
        current “topographic__elevation” field so that it will act as the initial 
        topographic condition the next time “run_one_step” is called. 

        Each time “run_one_step” is called, the component stores the current model 
        time, mean / standard deviation topographic change rate, and mean / standard 
        deviation erosion rate within appended arrays that can be accessed later. 

        The component provides functions to return, plot, and map all the calculated 
        values and arrays mentioned above. See the associated tutorial notebook 
        for more details.

    Case: “bedrock__and__soil” = True

        This mode should be used when the grid differentiates between bedrock and 
        soil (i.e. when the grid contains “bedrock__elevation” and “soil__depth” 
        fields). The component will verify that these two fields exist and that 
        (“bedrock__elevation” +  “soil__depth” = “topographic__elevation”). If 
        either of these conditions are not met, the component will display an 
        error. 

        This mode operates similarly to the other mode, but it also tracks elevational 
        / erosional changes on the bedrock surface and soil layer. See the associated 
        tutorial notebook for more details.

    Examples
    --------
    See associated notebook.
    
    """
    
    _name = "ErosionElevationTracker"
    
    _info = {
        "topographic__elevation": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m / yr",
            "mapping": "node",
            "doc": "Elevational change per unit time",
        },
    }
    
    _unit_agnostic = True
    
    # Initiate component
    def __init__(self, 
                 grid,
                 bedrock__and__soil = False
                 ):
        
        # Initiate grid
        super().__init__(grid)
        
        # Record status of bedrock__and__soil
        self._bedrock__and__soil = bedrock__and__soil
        
        # Case: bedrock__and__soil == False
        if bedrock__and__soil == False: 
            
            # Store initial topography
            self._elevation0 = np.copy(self.grid.at_node["topographic__elevation"])
            
            # Create empty variables for future appending
            self._ti = []
            self._dzdt_mean = []
            self._dzdt_std = []
            self._erosionrate_mean = []
            self._erosionrate_std = []
            
        # Case: bedrock__and__soil == True        
        if bedrock__and__soil == True:         
            
            # # Check whether grid contains "bedrock__elevation" field
            # if "bedrock__elevation" in grid.at_node.keys():
            #     self._bedrock__elevation0 = np.copy(self.grid.at_node["bedrock__elevation"])
            # else:
            #     message = (
            #         "ErosionElevationTracker component requires grid to contain "
            #         "the bedrock__elevation field when bedrock__and__soil "
            #         "= True) "
            #         )
            #     raise KeyError(message)
                
            # Check whether grid contains "soil__depth" field
            if "soil__depth" in grid.at_node.keys():
                self._soil__depth0 = np.copy(self.grid.at_node["soil__depth"])
            else:
                message = (
                    "ErosionElevationTracker component requires grid to contain "
                    "the soil__depth field when bedrock__and__soil "
                    "= True) "
                    )
                raise KeyError(message)
                
            # Check consistency of bedrock, soil and topogarphic elevation fields
            message = (
                "The sum of bedrock__elevation and soil__depth should be equal to topographic__elevation" 
                )
            np.testing.assert_almost_equal(
                grid.at_node["bedrock__elevation"] + grid.at_node["soil__depth"],
                grid.at_node["topographic__elevation"],
                decimal=5,
                err_msg=message,
                )
            
            # Store initial bedrock elevation, soil depth, and topography
            self._bedelev0 = np.copy(self.grid.at_node["bedrock__elevation"])
            self._soildepth0 = np.copy(self.grid.at_node["soil__depth"])
            self._elevation0 = np.copy(self.grid.at_node["topographic__elevation"])
            
            # Create empty variables for future appending
            self._ti = []
            self._dbedelevdt_mean = []
            self._dbedelevdt_std = []
            self._dsoildepthdt_mean = []
            self._dsoildepthdt_std = []
            self._dzdt_mean = []
            self._dzdt_std = []
            self._erosionrate_mean = []
            self._erosionrate_std = []

    # run_one_step           
    def run_one_step(self, 
                     dt,
                     uplift = 0,    # length, not velocity
                     ):
        
        # Case: bedrock__and__soil == False
        if self._bedrock__and__soil == False:  
        
            # Calculate changes since initiation or previous run_one_step
            self._elevation1 = np.copy(self.grid.at_node["topographic__elevation"])
            self._dz = self._elevation1 - self._elevation0
            self._dzdt = self._dz / dt
            self._erosion = uplift - self._dz
            self._erosionrate = self._erosion / dt
            
            # Reset initial topography
            self._elevation0 = np.copy(self.grid.at_node["topographic__elevation"])
        
            # Calculate mean and std dzdt and erosionrate since initiation or previous run_one_step
            dzdt_mean = np.mean(self._dzdt)
            dzdt_std = np.std(self._dzdt)
            erosionrate_mean = np.mean(self._erosionrate)
            erosionrate_std = np.std(self._erosionrate)
            
            # Store current mean and std dzdt and erosionrate with their associated time step
            # if self._ti == []:
            if len(self._ti) == 0:
                self._ti = np.append(self._ti, 
                                     dt
                                     )
            else:
                self._ti = np.append(self._ti, 
                                     self._ti[-1] + dt
                                     )
            self._dzdt_mean = np.append(self._dzdt_mean, 
                                        dzdt_mean
                                        )
            self._dzdt_std = np.append(self._dzdt_std, 
                                       dzdt_std
                                       )
            self._erosionrate_mean = np.append(self._erosionrate_mean, 
                                        erosionrate_mean
                                        )
            self._erosionrate_std = np.append(self._erosionrate_std, 
                                       erosionrate_std
                                       )
                
        # Case: bedrock__and__soil == True
        if self._bedrock__and__soil == True:   
            
            # # Check consistency of bedrock, soil and topogarphic elevation fields
            # message = (
            #     "The sum of bedrock__elevation and soil__depth should be equal to topographic__elevation" 
            #     )
            # np.testing.assert_almost_equal(
            #     self._grid.at_node["bedrock__elevation"] + self._grid.at_node["soil__depth"],
            #     self._grid.at_node["topographic__elevation"],
            #     decimal=5,
            #     err_msg=message,
            #     )

            # Calculate changes since initiation or previous run_one_step
            self._bedelev1 = np.copy(self.grid.at_node["bedrock__elevation"])
            self._soildepth1 = np.copy(self.grid.at_node["soil__depth"])
            self._elevation1 = np.copy(self.grid.at_node["topographic__elevation"])
            self._dbedelev = self._bedelev1 - self._bedelev0
            self._dsoildepth = self._soildepth1 - self._soildepth0
            self._dz = self._elevation1 - self._elevation0
            self._dbedelevdt = self._dbedelev / dt
            self._dsoildepthdt = self._dsoildepth / dt
            self._dzdt = self._dz / dt
            self._erosion = uplift - self._dz
            self._erosionrate = self._erosion / dt
            
            # Reset initial bedrock elevation, soil depth, and topograpghy
            self._bedelev0 = np.copy(self.grid.at_node["bedrock__elevation"])
            self._soildepth0 = np.copy(self.grid.at_node["soil__depth"])
            self._elevation0 = np.copy(self.grid.at_node["topographic__elevation"])             
        
            # Calculate mean and std d(metrics) since initiation or previous run_one_step
            dbedelevdt_mean = np.mean(self._dbedelevdt)
            dbedelevdt_std = np.std(self._dbedelevdt)
            dsoildepthdt_mean = np.mean(self._dsoildepthdt)
            dsoildepthdt_std = np.std(self._dsoildepthdt)
            dzdt_mean = np.mean(self._dzdt)
            dzdt_std = np.std(self._dzdt)   
            erosionrate_mean = np.mean(self._erosionrate)
            erosionrate_std = np.std(self._erosionrate)
        
            # # Store current mean and std dz / dt with their associated time step
            # if self._ti == []:
            #     self._ti = np.append(self._ti, 
            #                          dt
            #                          )
            # else:
            #     self._ti = np.append(self._ti, 
            #                          self._ti[-1] + dt
            #                          )
            # self._dbedelevdt_mean = np.append(self._dbedelevdt_mean,
            #                                   dbedelevdt_mean
            #                                   )
            # self._dbedelevdt_std = np.append(self._dbedelevdt_std,
            #                                   dbedelevdt_std
            #                                   )
            # self._dsoildepthdt_mean = np.append(self._dsoildepthdt_mean,
            #                                   dsoildepthdt_mean
            #                                   )
            # self._dsoildepthdt_std = np.append(self._dsoildepthdt_std,
            #                                   dsoildepthdt_std
            #                                   )
            # self._dzdt_mean = np.append(self._dzdt_mean, 
            #                             dzdt_mean
            #                             )
            # self._dzdt_std = np.append(self._dzdt_std, 
            #                            dzdt_std
            #                            )    
            # self._erosionrate_mean = np.append(self._erosionrate_mean, 
            #                             erosionrate_mean
            #                             )
            # self._erosionrate_std = np.append(self._erosionrate_std, 
            #                            erosionrate_std
            #                            )
       
    # Return data functions
    def return_dbedelev(self):
        return self._dbedelev
    
    def return_dbedelevdt(self):
        return self._dbedelevdt
    
    def return_dbedelevdt_mean(self):
        return self._dbedelevdt_mean
    
    def return_dbedelevdt_std(self):
        return self._dbedelevdt_std
    
    def return_dsoildepth(self):
        return self._dsoildepth

    def return_dsoildepthdt(self):
        return self._dsoildepthdt
    
    def return_dsoildepthdt_mean(self):
        return self._dsoildepthdt_mean
    
    def return_dsoildepthdt_std(self):
        return self._dsoildepthdt_std    
    
    def return_dz(self):
        return self._dz
    
    def return_dzdt(self):
        return self._dzdt
    
    def return_dzdt_mean(self):
        return self._dzdt_mean
    
    def return_dzdt_std(self):
        return self._dzdt_std
    
    def return_erosion(self):
        return self._erosion
    
    def return_erosionrate(self):
        return self._erosionrate
    
    def return_erosionrate_mean(self):
        return self._erosionrate_mean
    
    def return_erosionrate_std(self):
        return self._erosionrate_std

    # Mapping / plotting functions
    def map_dbedelev(self,  
                **kwds):
        imshow_grid(self.grid, 
                    self._dbedelev, 
                    cmap = 'BrBG',
                    symmetric_cbar = True,
                    **kwds)
        plt.title('d(bed elevation)')
    
    def map_dbedelevdt(self, 
                  **kwds):
        imshow_grid(self.grid, 
                    self._dbedelevdt, 
                    cmap = 'BrBG', 
                    symmetric_cbar = True,
                    **kwds)
        plt.title('d(bed elevation) / dt')
    
    def plot_dbedelevdt_mean(self, 
                       **kwds):
        plt.plot(self._ti, 
                 self._dbedelevdt_mean)
        plt.xlabel('time')
        plt.ylabel('d(mean bed elevation) / dt')
    
    def errorbar_dbedelevdt_mean(self, 
                           **kwds):
        plt.errorbar(self._ti, 
                     self._dbedelevdt_mean, 
                     self._dbedelevdt_std, 
                     zorder = 1,
                     **kwds)
        plt.plot(self._ti,
                 self._dbedelevdt_mean,
                 zorder = 2,
                 **kwds)
        plt.xlabel('time')
        plt.ylabel('d(mean bed elevation) / dt')
    
    def map_dsoildepth(self,  
                **kwds):
        imshow_grid(self.grid, 
                    self._dsoildepth, 
                    cmap = 'BrBG', 
                    symmetric_cbar = True,
                    **kwds)
        plt.title('d(soil depth)')
    
    def map_dsoildepthdt(self, 
                  **kwds):
        imshow_grid(self.grid, 
                    self._dsoildepthdt, 
                    cmap = 'BrBG', 
                    symmetric_cbar = True,
                    **kwds)
        plt.title('d(soil depth) / dt')
    
    def plot_dsoildepthdt_mean(self, 
                       **kwds):
        plt.plot(self._ti, 
                 self._dsoildepthdt_mean)
        plt.xlabel('time')
        plt.ylabel('d(mean soil depth) / dt')
    
    def errorbar_dsoildepthdt_mean(self, 
                           **kwds):
        plt.errorbar(self._ti, 
                     self._dsoildepthdt_mean, 
                     self._dsoildepthdt_std, 
                     zorder = 1,
                     **kwds)
        plt.plot(self._ti,
                 self._dsoildepthdt_mean,
                 zorder = 2,
                 **kwds)
        plt.xlabel('time')
        plt.ylabel('d(mean soil depth) / dt')
    
    def map_dz(self,  
                **kwds):
        imshow_grid(self.grid, 
                    self._dz, 
                    cmap = 'seismic_r', 
                    symmetric_cbar = True,
                    **kwds)
        plt.title('dz')
    
    def map_dzdt(self, 
                  **kwds):
        imshow_grid(self.grid, 
                    self._dzdt, 
                    cmap = 'seismic_r', 
                    symmetric_cbar = True,
                    **kwds)
        plt.title('dz / dt')
    
    def plot_dzdt_mean(self, 
                       **kwds):
        plt.plot(self._ti, 
                 self._dzdt_mean)
        plt.xlabel('time')
        plt.ylabel('d(mean elevation) / dt')
    
    def errorbar_dzdt_mean(self, 
                           **kwds):
        plt.errorbar(self._ti, 
                     self._dzdt_mean, 
                     self._dzdt_std, 
                     zorder = 1,
                     **kwds)
        plt.plot(self._ti,
                 self._dzdt_mean,
                 zorder = 2,
                 **kwds)
        plt.xlabel('time')
        plt.ylabel('d(mean elevation) / dt')
        
    def map_erosion(self,  
                **kwds):
        imshow_grid(self.grid, 
                    self._erosion, 
                    cmap = 'seismic', 
                    symmetric_cbar = True,
                    **kwds)
        plt.title('erosion')
    
    def map_erosionrate(self, 
                  **kwds):
        imshow_grid(self.grid, 
                    self._erosionrate, 
                    cmap = 'seismic', 
                    symmetric_cbar = True,
                    **kwds)
        plt.title('erosion rate')
    
    def plot_erosionrate_mean(self, 
                       **kwds):
        plt.plot(self._ti, 
                 self._erosionrate_mean)
        plt.xlabel('time')
        plt.ylabel('mean erosion rate')
    
    def errorbar_erosionrate_mean(self, 
                           **kwds):
        plt.errorbar(self._ti, 
                     self._erosionrate_mean, 
                     self._erosionrate_std, 
                     zorder = 1,
                     **kwds)
        plt.plot(self._ti,
                 self._erosionrate_mean,
                 zorder = 2,
                 **kwds)
        plt.xlabel('time')
        plt.ylabel('mean erosion rate')