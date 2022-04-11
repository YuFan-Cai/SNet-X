close all; clear; clc;

L = 1; 
Iteration = 0;
complex = 0;

if Iteration == 0
    hW = 10;
    hD = 3;
    alpha = 0.92;
    T = 0.2;
    nbit = 25;
else
    hW = (10);
    hD = (3);
    alpha = 0.92;
    T = 0.2;
    nbit = (25);
end

Test_box = {'../../Data/Test/Virtual-SAR/Test-A.tiff';
            '../../Data/Test/Virtual-SAR/Test-B.tiff';
            '../../Data/Test/Sentinel-1A/Test-VH.tiff';
            '../../Data/Test/CP-SAR/Test-RR.tiff';
            '../../Data/Test/ERS-1/Test-VV.tiff'};

Save_box = {'./PPB-Virtual-SAR-A.tiff';
            './PPB-Virtual-SAR-B.tiff';
            './PPB-Sentinel-1A.tiff';
            './PPB-CP-SAR.tiff';
            './PPB-ERS-1.tiff'};

Num = 5;
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

s = zeros(size(img));
if complex ~= 0
    for k = 1:L
        s = s + abs(randn(size(img)) + 1i * randn(size(img))).^2 / 2;
    end
    ima_nse = img .* sqrt(s / L);
else
    ima_nse = img;
end
   
tic;
out = ppb_nakagami(ima_nse, L, hW, hD, alpha, T, nbit);
out = max(out, 0);
toc;
t.write(out);
t.close