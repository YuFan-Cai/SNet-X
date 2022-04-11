close all; clear; clc;

L = 1;

Test_box = {'../../Data/Test/Virtual-SAR/Test-A.tiff';
            '../../Data/Test/Virtual-SAR/Test-B.tiff';
            '../../Data/Test/Sentinel-1A/Test-VH.tiff';
            '../../Data/Test/CP-SAR/Test-RR.tiff';
            '../../Data/Test/ERS-1/Test-VV.tiff'};

Save_box = {'./FANS-Virtual-SAR-A.tiff';
            './FANS-Virtual-SAR-B.tiff';
            './FANS-Sentinel-1A.tiff';
            './FANS-CP-SAR.tiff';
            './FANS-ERS-1.tiff'};

Num = 1;
Data_path = char(Test_box(Num));
img = imread(Data_path);
img = double(img);

Save_path = char(Save_box(Num));
t = Tiff(Save_path,'w');
tagstruct.ImageLength = size(img,1);
tagstruct.ImageWidth = size(img,2);
tagstruct.Photometric = 1;
tagstruct.BitsPerSample = 64;
tagstruct.SamplesPerPixel = 1;
tagstruct.RowsPerStrip = 16;
tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
tagstruct.Software = 'MATLAB'; 
tagstruct.SampleFormat = 3;
t.setTag(tagstruct)
   
tic;
out = FANS(img, L);
out = max(out, 0);
toc;
t.write(out);
t.close