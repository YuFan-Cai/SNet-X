close all; clear; clc;

L = 1; 

Test_box = {'../../Data/Test/Virtual-SAR/Test-A.tiff';
            '../../Data/Test/Virtual-SAR/Test-B.tiff';
            '../../Data/Test/Sentinel-1A/Test-VH.tiff';
            '../../Data/Test/CP-SAR/Test-RR.tiff';
            '../../Data/Test/ERS-1/Test-VV.tiff'};

Save_box = {'./RABASAR-Virtual-SAR-A.tiff';
            './RABASAR-Virtual-SAR-B.tiff';
            './RABASAR-Sentinel-1A-1.tiff';
            './RABASAR-CP-SAR.tiff';
            './RABASAR-ERS-1.tiff'};

Num = 3;
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
[M, N, T] = size(img);
thrs = bw_thresholds(M, N, L, 7, .92);
[out, u_hat_si, L_hat_si, u_hat_dsi, L_hat_dsi] = rabasar(img, L, T, 'thrs', thrs);
out = max(out, 0);
toc;
t.write(out);
t.close