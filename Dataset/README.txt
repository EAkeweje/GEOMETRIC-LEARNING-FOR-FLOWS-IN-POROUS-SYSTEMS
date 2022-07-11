Content of archive:
1) vel_fields - folder with dataset of velocity fields. Dimensionless, LB units. Steady-state reached by RMSE convergence for interval of 5000 steps. All files are represented in .dat format, which can be easily opened by any notepad. Index at the start of the filename connects the velocity field to the inlet velocity profile from "vps" folder. You should take in mind the rotation of obtained fields (watch Rotation.png)

2) vps - folder contains .txt files, which represent the array of X-components of velocities for inlet boundary. Array consist of 256 points (length of the side of the sample in voxels). Velocities are dimensionless, LB units. Index at the end of the filename connects to the velocity field from "vel_fields" folder.

3) 1.png - Image of the porous media sample in png black/white format. You should take in mind the rotation of axes (watch Rotation.png)

4)Rotation.png - Scheme, showing the swap of axes between velocity fields and image of the porous media sample.

5) Processing_vel_fields.png - screenshot of "howto" process, upload and plot the velocity fields by each component. 0 component of array is X-component, 1 - Y-component.