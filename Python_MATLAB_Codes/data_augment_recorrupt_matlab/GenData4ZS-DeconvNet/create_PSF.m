cd 'C:\Users\Admin\Git\repo\ZS-DeconvNet\Python_MATLAB_Codes\data_augment_recorrupt_matlab';
addpath(genpath('./XxUtils'));

dxy = 92.6; % lateral sampling, in (nm)
dz = 92.6; % axial sampling, in (nm)
SizeXY = 27; % lateral pixel number of PSF
SizeZ = 13; % axial pixel number of PSF
lamd = 525; % emission wavelength, in (nm)
NA = 1.1; % numerical aperture
RI = 1.3; % refractive index
FileName = './GenData4ZS-DeconvNet/SimulatedPSF.tif';
XxPSFGenerator(dxy,dz,SizeXY,SizeZ,lamd,NA,RI,FileName);