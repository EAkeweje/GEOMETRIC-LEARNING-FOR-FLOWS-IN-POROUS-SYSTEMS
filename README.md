# GEOMETRIC LEARNING FOR FLOWS IN POROUS SYSTEMS

## Aim and Objectives
Consider a complex network (porous media, river basin, pipeline network) where fluid flow is governed by strict laws of physics and boundary condition unknown or difficult to calculate. Assuming either historical data is available with good precision or a limited number of snapshots can be calculated. The aim of this study is to reconstruct the state of fluid flow from real-time measurements obtained from a limited number of sensors located at particular points in the complex network. The objectives of this study are therefore:
- Reconstruct of the state of the system (from experimental measurements),
- Optimize the location of the gauges.

## About Dataset
1) vel_fields - folder with dataset of velocity fields. Dimensionless, LB units. Steady-state reached by RMSE convergence for interval of 5000 steps. All files are represented in .dat format, which can be easily opened by any notepad. Index at the start of the filename connects the velocity field to the inlet velocity profile from "vps" folder. You should take in mind the rotation of obtained fields (watch Rotation.png)

2) vps - folder contains .txt files, which represent the array of X-components of velocities for inlet boundary. Array consist of 256 points (length of the side of the sample in voxels). Velocities are dimensionless, LB units. Index at the end of the filename connects to the velocity field from "vel_fields" folder.

3) 1.png - Image of the porous media sample in png black/white format. You should take in mind the rotation of axes (watch Rotation.png)

4)Rotation.png - Scheme, showing the swap of axes between velocity fields and image of the porous media sample.

5) Processing_vel_fields.png - screenshot of "howto" process, upload and plot the velocity fields by each component. 0 component of array is X-component, 1 - Y-component.
