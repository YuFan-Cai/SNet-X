%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  M Estimator (homogeneity, ENL, Mean)
%  Reproducible research for article: 
%  L. Gomez, R. Ospina, A. C. Frery
%  "Unassisted quantitative evaluation of despeckling filters"
%  Work by  L. Gomez, R. Ospina, A. C. Frery; codified to Matlab by L. Gomez 
%  Run well on Matlab 2014a
%  February 2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Please use it freely and cite the reference paper mentioned above.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function varargout = ENL_Mean_Estimator(varargin)
% ENL_MEAN_ESTIMATOR M-file for ENL_Mean_Estimator.fig
%      ENL_MEAN_ESTIMATOR, by itself, creates a new ENL_MEAN_ESTIMATOR or raises the existing
%      singleton*.
%
%      H = ENL_MEAN_ESTIMATOR returns the handle to a new ENL_MEAN_ESTIMATOR or the handle to
%      the existing singleton*.
%
%      ENL_MEAN_ESTIMATOR('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in ENL_MEAN_ESTIMATOR.M with the given input arguments.
%
%      ENL_MEAN_ESTIMATOR('Property','Value',...) creates a new ENL_MEAN_ESTIMATOR or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before ENL_Mean_Estimator_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to ENL_Mean_Estimator_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help ENL_Mean_Estimator

% Last Modified by GUIDE v2.5 31-Mar-2017 20:34:37

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @ENL_Mean_Estimator_OpeningFcn, ...
                   'gui_OutputFcn',  @ENL_Mean_Estimator_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes just before ENL_Mean_Estimator is made visible.
function ENL_Mean_Estimator_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to ENL_Mean_Estimator (see VARARGIN)


%Default values
handles.radiobutton5 = 0; %Intensity data
handles.radiobutton6 = 1; %Amplitude data 
tmp = '1';set(handles.edit1,'String',tmp);
handles.edit1 = 1; %ENL default value
tmp = '25';set(handles.edit6,'String',tmp);
handles.edit6 = 25; %Mask size Haralicks default value
tmp = '10';set(handles.edit21,'String',tmp);
handles.edit21 = 10; %Number random tests
handles.edit22 = 5; %ENL Tolerance default value
handles.edit24 = 25; %Mask size for ENL & Mean default value
tmp = '32';set(handles.edit27,'String',tmp);
handles.edit27 = 32; %Number gray levels Haralicks default value
tmp = '0.5';set(handles.edit28,'String',tmp);
handles.edit28 = 0.5; %Ponderation default value
tmp = '4';set(handles.edit31,'String',tmp);
handles.edit31 = 8; %Min. Rows/cols default value
tmp = '16';set(handles.edit32,'String',tmp);
handles.edit32 = 16; %Max. Rows/cols default value
tmp = '2';set(handles.edit33,'String',tmp);
handles.edit33 = 2; %Rows/cols step default value
tmp = '0.001';set(handles.edit34,'String',tmp);
handles.edit34 = 0.001; %Min. PFA default value
tmp = '0.005';set(handles.edit35,'String',tmp);
handles.edit35 = 0.005; %Max. PFA default value
tmp = '0.001';set(handles.edit36,'String',tmp);
handles.edit36 = 0.001; %PFA step default value
tmp = '100';set(handles.edit48,'String',tmp);
handles.edit48 = 100; % k coefficient (delta_h) default value
handles.checkbox1 = 0; % Meyer wavelet transform
handles.checkbox2 = 0; % DCT wavelet transform
handles.checkbox3 = 0; % Haar wavelet transform
handles.checkbox4 = 1; % Daub2 wavelet transform
handles.checkbox6 = 0; % Daub3 wavelet transform
handles.checkbox7 = 0; % Daub4 wavelet transform
handles.checkbox8 = 0; % Bior1.3 wavelet transform
handles.checkbox9 = 0; % Bioir1.5 wavelet transform


tmp = '';set(handles.edit2,'String',tmp);
tmp = '';set(handles.edit3,'String',tmp);
tmp = '';set(handles.edit4,'String',tmp);
tmp = '';set(handles.edit8,'String',tmp);
tmp = '';set(handles.edit9,'String',tmp);
tmp = '';set(handles.edit10,'String',tmp);
tmp = '';set(handles.edit15,'String',tmp);
tmp = '';set(handles.edit16,'String',tmp);
tmp = '';set(handles.edit19,'String',tmp);
tmp = '';set(handles.edit20,'String',tmp);
tmp = '';set(handles.edit23,'String',tmp);
tmp = '';set(handles.edit25,'String',tmp);
tmp = '';set(handles.edit26,'String',tmp);
tmp = '';set(handles.edit29,'String',tmp);
tmp = '';set(handles.edit30,'String',tmp);
tmp = '';set(handles.edit37,'String',tmp);
tmp = '';set(handles.edit40,'String',tmp);
tmp = '';set(handles.edit41,'String',tmp);
tmp = '';set(handles.edit43,'String',tmp);
tmp = '';set(handles.edit42,'String',tmp);
tmp = '';set(handles.edit44,'String',tmp);
tmp = '';set(handles.edit45,'String',tmp);
tmp = '';set(handles.edit46,'String',tmp);
tmp = '';set(handles.edit47,'String',tmp);

% Choose default command line output for ENL_Mean_Estimator
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes ENL_Mean_Estimator wait for user response (see UIRESUME)
% uiwait(handles.figure1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Outputs from this function are returned to the command line.
function varargout = ENL_Mean_Estimator_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
input_images = handles.popupmenu1;
I = input_images.Noisy_image;
escala_display=mean(I(:))*3;figure(),imshow(imresize(I,1),[0,escala_display]);
title('Noisy Image');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
input_images = handles.popupmenu1;
I = input_images.Filtered_image;
escala_display=mean(I(:))*3;figure(),imshow(imresize(I,1),[0,escala_display]);
title('Filtered Image');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
input_images = handles.popupmenu1;
I = input_images.Ratio_image;
escala_display=mean(I(:))*3;figure(),imshow(imresize(I,1),[0,escala_display]);
title('Ratio Image');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
input_images = handles.pushbutton9;
I = input_images.H0;
escala_display=mean(I(:))*3;figure(),imshow(imresize(I,1),[0,escala_display]);
title('H0');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in pushbutton9.
function pushbutton9_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

input_images = handles.popupmenu1;
ratio = input_images.Ratio_image;
[m n] = size(ratio);
mn = m*n;
ENL = handles.edit1;
option_intensity = handles.radiobutton5;
option_amplitude = handles.radiobutton6;


if(option_intensity == 1) %intensity
     display('Intensity');
    shape = ENL;
    scale = 1 / shape;
    Y = gamrnd(shape,scale,m,n);    
    roi= Y(1:100,1:100);
    m_Y = mean2(roi);
    s_Y = std2(roi);
    ENL_Y= m_Y^2/s_Y^2  
    set(handles.edit2,'String',m_Y);
    set(handles.edit3,'String',s_Y);
    set(handles.edit4,'String',ENL_Y);
elseif(option_amplitude == 1) % amplitude
    display('Amplitude');
   sigma = sqrt(2/pi);
   s = zeros(m,n);
   for k = 1: ENL
     u = rand(m,n);
      s = s + sqrt(-2*sigma^2*log(1-u));
   end
   Y = (s/ ENL);
   roi= Y(1:100,1:100);
   m_Y = mean2(roi);
   s_Y = std2(roi);
   ENL_Y =   0.2732/(s_Y/m_Y)^2;
    set(handles.edit2,'String',m_Y);
    set(handles.edit3,'String',s_Y);
    set(handles.edit4,'String',ENL_Y);   
end
  axes(handles.axes13);
  escala_display=mean(Y(:))*3; imshow(imresize(Y,1),[0,escala_display]);
 H0_data = struct('H0', Y,'Mean_H0', m_Y, 'Std_H0', s_Y, 'ENL_H0', ENL_Y);
handles.pushbutton9 = H0_data;
guidata(hObject,handles)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on selection change in popupmenu1.
function popupmenu1_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns popupmenu1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu1
%Default values
tmp = '';set(handles.edit2,'String',tmp);
tmp = '';set(handles.edit3,'String',tmp);
tmp = '';set(handles.edit4,'String',tmp);
tmp = '';set(handles.edit8,'String',tmp);
tmp = '';set(handles.edit9,'String',tmp);
tmp = '';set(handles.edit10,'String',tmp);
tmp = '';set(handles.edit13,'String',tmp);
tmp = '';set(handles.edit12,'String',tmp);
tmp = '';set(handles.edit14,'String',tmp);
tmp = '';set(handles.edit15,'String',tmp);
tmp = '';set(handles.edit16,'String',tmp);
tmp = '';set(handles.edit19,'String',tmp);
tmp = '';set(handles.edit20,'String',tmp);
tmp = '';set(handles.edit23,'String',tmp);
tmp = '';set(handles.edit25,'String',tmp);
tmp = '';set(handles.edit26,'String',tmp);
clc
ENL = handles.edit1
Mask_Size = handles.edit6


% Load data: it must be in \data
% The data file has:
%   - I (noisy image)
%   - F (filtered image)
% Ratio image is obtaines here as I / F
%   size(F) must be equal to size(ratio)

Path='..\data';
[FileName Path]=uigetfile({'*.mat'},'Load Input Data');

text = FileName;
cd data
load(text);
cd ..


% Check image sizes
[m_I n_I] = size(I);
[m_F n_F] = size(F);

if(m_F ~= m_I || n_F ~= n_I)
    display('Error: Size of Noisy Image and Filtered image must be equal.');
end
    
%Plot image
% Plot noisy  image
axes(handles.axes1);
escala_display=mean(I(:))*3;imshow(imresize(I,1),[0,escala_display]);
% Plot filtered image
axes(handles.axes9);
escala_display=mean(F(:))*3;imshow(imresize(F,1),[0,escala_display]);
% Obtain ratio image:
ratio = I./ F;
% Plot ratio image
axes(handles.axes10);
escala_display=mean(ratio(:))*3; imshow(imresize(ratio,1),[0,escala_display]);

input_images = struct('Noisy_image', I, 'Filtered_image',F, 'Ratio_image',ratio);
handles.popupmenu1=input_images;
guidata(hObject,handles);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes during object creation, after setting all properties.
function popupmenu1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in pushbutton10.
function pushbutton10_Callback(hObject, eventdata, handles)
global ENL
clc
input_images = handles.popupmenu1;
ratio = input_images.Ratio_image;
Noisy = input_images.Noisy_image;
[m n] = size(ratio);
%ratio =  imadjust(ratio);
 option_intensity = handles.radiobutton5;
 option_amplitude = handles.radiobutton6;

if(option_intensity == 1) %intensity mode
   option = 0;
    display('Intensity Mode');
elseif(option_amplitude == 1) %amplitude mode
    option = 1;
     display('Amplitude Mode');
end

%Haralick's homogeneity (only measured within non-homogeneous areas
offset_tmp= handles.edit6;
levels_Haralick = handles.edit27;
%%%%%%%%%%%%%%%%%%Haralick's homogeneity by using Matlab functions%%%%%%
% % All directions
offset = [0 offset_tmp; -offset_tmp offset_tmp; -offset_tmp 0; -offset_tmp -offset_tmp];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
escala_display=mean(ratio(:))*3;figure(10),imshow(imresize(ratio,1),[0,escala_display]);
%figure(10), imshow(ratio_enhanced)
glcm_data = graycomatrix(ratio,'NumLevels',levels_Haralick, 'Offset', offset,'Symmetric',true,'GrayLimits',[]);
% The average
tmp = glcm_data(:,:,1) + glcm_data(:,:,2) + glcm_data(:,:,3) + glcm_data(:,:,4);
glcm_average = tmp / 4;

% Haralick's descriptors for ratio image
Haralicks = GLCMFeatures(glcm_average);

% homogeneity 
homogeneity_ratio = Haralicks.homogeneity; 
%%%%%%%%%%%%%%%%%%Haralick's homogeneity by using Matlab functions%%%%%%
set(handles.edit8,'String',homogeneity_ratio);

%homogeneity for ratio disorderered
n_random_tests = handles.edit21;
homogeneity_disordered = zeros(n_random_tests,1);
[m n] = size(ratio);
mn = m*n;
II = ratio(:);
for ii = 1: n_random_tests
    r = randperm(mn);
    I_random = zeros(mn,1);
    for i = 1:numel(r)
        I_random(i) = II(r(i));
    end
    ratio_tmp = reshape(I_random, m,n);
    %escala_display=mean(ratio_tmp(:))*3;figure(12),imshow(imresize(ratio_tmp,1),[0,escala_display]);
    %%%%%%%%Haralick's homogeneity by using Matlab functions%%%%%
     % All directions
    glcm_data = graycomatrix(ratio_tmp,'NumLevels',levels_Haralick, 'Offset', offset,'Symmetric',true,'GrayLimits',[]);
    
    % The average
    tmp = glcm_data(:,:,1) + glcm_data(:,:,2) + glcm_data(:,:,3) +glcm_data(:,:,4);
    glcm_average = tmp / 4;
    
    % Haralick's descriptors for ratio image
    Haralicks = GLCMFeatures(glcm_average);
    
    % homogeneity
     homogeneity_disordered(ii) = Haralicks.homogeneity; 
     homogeneity_disordered
    %%%%%%%%Haralick's homogeneity by using Matlab functions%%%%%     
  end
homogeneity_disordered = mean(homogeneity_disordered); 
set(handles.edit15,'String',homogeneity_disordered);
delta_homogeneity = 100 * abs((homogeneity_ratio - homogeneity_disordered)/homogeneity_disordered); 
set(handles.edit16,'String',delta_homogeneity);

%ENL_M EAN estimator (residual)
% ENL Mean measured on Ratio but within homogeneous areas within the Noisy
% image
ENL = handles.edit1;
mask_ENL = handles.edit24;
ENL_tol = handles.edit22;
ponderation = handles.edit28;

if(option ==0) Noisy_ENL = blkproc(Noisy,[mask_ENL mask_ENL],@estimator_ENL_intensity);
else Noisy_ENL = blkproc(Noisy,[mask_ENL mask_ENL],@estimator_ENL_amplitude);
end
[mm nn] = size(Noisy_ENL);
Noisy_ENL = Noisy_ENL(1:mm-1,1:nn-1);
[m n] = size(Noisy_ENL);
%look for homogeneous areas
inf_tol = ENL - ENL * ENL_tol / 100
sup_tol = ENL + ENL * ENL_tol / 100
Noisy_homogeneous =(Noisy_ENL >= inf_tol) & (Noisy_ENL <=  sup_tol);

if(option ==0) Ratio_ENL = blkproc(ratio,[mask_ENL mask_ENL],@estimator_ENL_intensity);
else Ratio_ENL = blkproc(ratio,[mask_ENL mask_ENL],@estimator_ENL_amplitude);
end
[mm nn] = size(Ratio_ENL);
Ratio_ENL = Ratio_ENL(1:mm-1,1:nn-1);

Mean_Ratio = blkproc(ratio, [mask_ENL mask_ENL], @Mean_image);
[mm nn] = size(Mean_Ratio);
Mean_Ratio = Mean_Ratio(1:mm-1,1:nn-1);

TOTAL_ENL_Mean_ratio = 0.0;
r_ENL = 0;
r_mu = 0;
for i = 1: m
    for j = 1: n
        if(Noisy_homogeneous(i,j) == 1)
            res_ENL = abs(Noisy_ENL(i,j) - Ratio_ENL(i,j))/Noisy_ENL(i,j);
            res_mu =  abs(1 - Mean_Ratio(i,j));
            TOTAL_ENL_Mean_ratio = TOTAL_ENL_Mean_ratio + (res_ENL + res_mu) * 100/2;
            r_ENL = r_ENL + res_ENL;
            r_mu = r_mu + res_mu;
         end
    end
end

total_homogeneous_area = numel(find(Noisy_homogeneous > 0));
TOTAL_ENL_Mean_ratio = TOTAL_ENL_Mean_ratio/total_homogeneous_area; 
set(handles.edit23,'String',TOTAL_ENL_Mean_ratio);
set(handles.edit25,'String',total_homogeneous_area);
k = handles.edit48; %coefficiento to scale delta_h
set(handles.edit26,'String',(TOTAL_ENL_Mean_ratio * ponderation  + k * delta_homogeneity * (1 - ponderation)));

%to control (not used in the calculus)
r_ENL = 100* r_ENL / total_homogeneous_area
r_mu = 100* r_mu / total_homogeneous_area

ENL_Mean_Data = struct('Homogeneity', homogeneity_ratio,'Homogeneity_disordered', homogeneity_disordered, 'Variation',delta_homogeneity, 'ENL_MEAN', TOTAL_ENL_Mean_ratio );
handles.pushbutton10 = ENL_Mean_Data;
guidata(hObject,handles)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edit8_Callback(hObject, eventdata, handles)
% hObject    handle to edit8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit8 as text
%        str2double(get(hObject,'String')) returns contents of edit8 as a double


% --- Executes during object creation, after setting all properties.
function edit8_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in pushbutton14.
function pushbutton14_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edit9_Callback(hObject, eventdata, handles)
% hObject    handle to edit9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit9 as text
%        str2double(get(hObject,'String')) returns contents of edit9 as a double


% --- Executes during object creation, after setting all properties.
function edit9_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in pushbutton15.
function pushbutton15_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton15 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edit10_Callback(hObject, eventdata, handles)
% hObject    handle to edit10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit10 as text
%        str2double(get(hObject,'String')) returns contents of edit10 as a double
% hObject    handle to pushbutton10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global ENL
clc
input_images = handles.popupmenu1;
ratio = input_images.Ratio_image;
Noisy = input_images.Noisy_image;
[m n] = size(ratio);
ratio =  imadjust(ratio);
 option_intensity = handles.radiobutton5;
 option_amplitude = handles.radiobutton6;


if(option_intensity == 1) %intensity mode
   option = 0;
    display('Intensity Mode');
elseif(option_amplitude == 1) %amplitude mode
    option = 1;
     display('Amplitude Mode');
end

%Haralick's homogeneity (only measured within non-homogeneous areas
offset_tmp= handles.edit6;
levels_Haralick = handles.edit27;
%%%%%%%%%%%%%%%%%%Haralick's homogeneity by using Matlab functions%%%%%%
% % All directions
offset = [0 offset_tmp; -offset_tmp offset_tmp; -offset_tmp 0; -offset_tmp -offset_tmp];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%escala_display=mean(ratio(:))*3;figure(10),imshow(imresize(ratio,1),[0,escala_display]);
glcm_data = graycomatrix(ratio,'NumLevels',levels_Haralick, 'Offset', offset,'Symmetric',true,'GrayLimits',[]);
% The average
tmp = glcm_data(:,:,1) + glcm_data(:,:,2) + glcm_data(:,:,3) + glcm_data(:,:,4);
glcm_average = tmp / 4;

% Haralick's descriptors for ratio image
Haralicks = GLCMFeatures(glcm_average);

% homogeneity 
homogeneity = Haralicks.homogeneity;
%%%%%%%%%%%%%%%%%Haralick's homogeneity by using Matlab functions%%%%%%
set(handles.edit8,'String',homogeneity);

%homogeneity for ratio disorderered
n_random_tests = handles.edit21;
homogeneity_disordered = zeros(n_random_tests,1);
[m n] = size(ratio);
mn = m*n;
II = ratio(:);
for ii = 1: n_random_tests
    r = randperm(mn);
    I_random = zeros(mn,1);
    for i = 1:numel(r)
        I_random(i) = II(r(i));
    end
    ratio_tmp = zeros(m,n);
    %make the matrix
    ratio_tmp = reshape(ratio_tmp,m,n);
   % escala_display=mean(ratio_tmp(:))*3;figure(12),imshow(imresize(ratio_tmp,1),[0,escala_display]);
    %%%%%%%%Haralick's homogeneity by using Matlab functions%%%%%
     % All directions
    glcm_data = graycomatrix(ratio_tmp,'NumLevels',levels_Haralick, 'Offset', offset,'Symmetric',true,'GrayLimits',[]);
    
    % The average
    tmp = glcm_data(:,:,1) + glcm_data(:,:,2) + glcm_data(:,:,3) +glcm_data(:,:,4);
    glcm_average = tmp / 4;
    
    % Haralick's descriptors for ratio image
    Haralicks = GLCMFeatures(glcm_average);
    
    % homogeneity
    homogeneity_disordered(ii) = Haralicks.homogeneity; 
    Haralicks.homogeneity
    %%%%%%%%Haralick's homogeneity by using Matlab functions%%%%%     
  end
homogeneity_disordered = mean(homogeneity_disordered); %old
set(handles.edit15,'String',homogeneity_disordered);
delta_homogeneity = 1 * abs((homogeneity - homogeneity_disordered)/homogeneity); 
set(handles.edit16,'String',delta_homogeneity);


%ENL_M EAN estimator (residual)
% ENL Mean measured on Ratio but within homogeneous areas within the Noisy
% image
ENL = handles.edit1;
mask_ENL = handles.edit24;
ENL_tol = handles.edit22;
ponderation = handles.edit28;

if(option ==0) Noisy_ENL = blkproc(Noisy,[mask_ENL mask_ENL],@estimator_ENL_intensity);
else Noisy_ENL = blkproc(Noisy,[mask_ENL mask_ENL],@estimator_ENL_amplitude);
end
[mm nn] = size(Noisy_ENL);
Noisy_ENL = Noisy_ENL(1:mm-1,1:nn-1);
[m n] = size(Noisy_ENL);
%look for homogeneous areas
inf_tol = ENL - ENL * ENL_tol / 100;
sup_tol = ENL + ENL * ENL_tol / 100;
Noisy_homogeneous =(Noisy_ENL > inf_tol) & (Noisy_ENL <  sup_tol);

if(option ==0) Ratio_ENL = blkproc(ratio,[mask_ENL mask_ENL],@estimator_ENL_intensity);
else Ratio_ENL = blkproc(ratio,[mask_ENL mask_ENL],@estimator_ENL_amplitude);
end
[mm nn] = size(Ratio_ENL);
Ratio_ENL = Ratio_ENL(1:mm-1,1:nn-1);

Mean_Ratio = blkproc(ratio, [mask_ENL mask_ENL], @Mean_image);
[mm nn] = size(Mean_Ratio);
Mean_Ratio = Mean_Ratio(1:mm-1,1:nn-1);

TOTAL_ENL_Mean_ratio = 0.0;
for i = 1: m
    for j = 1: n
        if(Noisy_homogeneous(i,j) == 1)
            TOTAL_ENL_Mean_ratio = TOTAL_ENL_Mean_ratio + (abs(Noisy_ENL(i,j) - Ratio_ENL(i,j))/abs(Noisy_ENL(i,j))*100 + abs(1-Mean_Ratio(i,j))*100)/2; %revised (elative,tanto por 1)
        end
    end
end

total_homogeneous_area = numel(find(Noisy_homogeneous > 0));
TOTAL_ENL_Mean_ratio = TOTAL_ENL_Mean_ratio/total_homogeneous_area; 
set(handles.edit23,'String',TOTAL_ENL_Mean_ratio);
set(handles.edit25,'String',total_homogeneous_area);
set(handles.edit26,'String',(TOTAL_ENL_Mean_ratio * ponderation  + delta_homogeneity * (1 - ponderation)));


ENL_Mean_Data = struct('Homogeneity', homogeneity,'Homogeneity_disordered', homogeneity_disordered, 'Variation',delta_homogeneity, 'ENL_MEAN', TOTAL_ENL_Mean_ratio );
handles.pushbutton10 = ENL_Mean_Data;
guidata(hObject,handles)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes during object creation, after setting all properties.
function edit10_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edit2_Callback(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit2 as text
%        str2double(get(hObject,'String')) returns contents of edit2 as a double


% --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edit3_Callback(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit3 as text
%        str2double(get(hObject,'String')) returns contents of edit3 as a double


% --- Executes during object creation, after setting all properties.
function edit3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edit4_Callback(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit4 as text
%        str2double(get(hObject,'String')) returns contents of edit4 as a double


% --- Executes during object creation, after setting all properties.
function edit4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double
ENL = str2num(get(hObject,'String'));
if(ENL <= 0.0) errordlg('Error, ENL >= 0.');
end

handles.edit1 = ENL;
guidata(hObject,handles)


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edit6_Callback(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit6 as text
%        str2double(get(hObject,'String')) returns contents of edit6 as a double
mask_size = str2num(get(hObject,'String'));
if(mask_size < 3) errordlg('Error, Mask Size>= 3.');
end
tmp = mod(mask_size,2);
if(tmp == 0) 
    errordlg('Error,mask for estimator needs to have uneven size (eg. 5x5).');
end
handles.edit6 = mask_size;
guidata(hObject,handles)


% --- Executes during object creation, after setting all properties.
function edit6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in pushbutton16.
function pushbutton16_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton16 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
images = handles.popupmenu1;
I = images.Noisy_image;

h = figure();
escala_display=mean(I(:))*3; imshow(imresize(I,1),[0,escala_display]);
title('Noisy Image');
rect = getrect;
roi = imcrop(I,rect);
close(h);

mode_1 = handles.radiobutton5;  % intensity
mode_2 = handles.radiobutton6;  % amplitude

mean_roi = mean2(roi);
std_roi = std2(roi);
if(mode_1 == 1) %intensity data
    ENL = mean_roi^2 /  std_roi^2;
      display('Intensity')
elseif mode_2 == 1; % amplitude data
    ENL =  0.2732 * mean_roi^2 /  std_roi^2;
      display('Amplitude')
end

set(handles.edit13,'String',mean_roi);
set(handles.edit14,'String',std_roi);
set(handles.edit12,'String',ENL);

% Noisy Image
I = images.Filtered_image;
roi = imcrop(I,rect);

mean_roi = mean2(roi);
std_roi = std2(roi);
if(mode_1 == 1) %intensity data
    ENL = mean_roi^2 /  std_roi^2;
      display('Intensity')
elseif mode_2 == 1; % amplitude data
    ENL =  0.2732 * mean_roi^2 /  std_roi^2;
      display('Amplitude')
end

set(handles.edit43,'String',mean_roi);
set(handles.edit44,'String',std_roi);
set(handles.edit42,'String',ENL);

% Ratio Image
I = images.Ratio_image;
roi = imcrop(I,rect);

mean_roi = mean2(roi);
std_roi = std2(roi);
if(mode_1 == 1) %intensity data
    ENL = mean_roi^2 /  std_roi^2;
      display('Intensity')
elseif mode_2 == 1; % amplitude data
    ENL =  0.2732 * mean_roi^2 /  std_roi^2;
      display('Amplitude')
end

set(handles.edit46,'String',mean_roi);
set(handles.edit47,'String',std_roi);
set(handles.edit45,'String',ENL);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edit12_Callback(hObject, eventdata, handles)
% hObject    handle to edit12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit12 as text
%        str2double(get(hObject,'String')) returns contents of edit12 as a double


% --- Executes during object creation, after setting all properties.
function edit12_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edit13_Callback(hObject, eventdata, handles)
% hObject    handle to edit13 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit13 as text
%        str2double(get(hObject,'String')) returns contents of edit13 as a double


% --- Executes during object creation, after setting all properties.
function edit13_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit13 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edit14_Callback(hObject, eventdata, handles)
% hObject    handle to edit14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit14 as text
%        str2double(get(hObject,'String')) returns contents of edit14 as a double


% --- Executes during object creation, after setting all properties.
function edit14_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edit15_Callback(hObject, eventdata, handles)
% hObject    handle to edit15 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit15 as text
%        str2double(get(hObject,'String')) returns contents of edit15 as a double


% --- Executes during object creation, after setting all properties.
function edit15_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit15 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edit16_Callback(hObject, eventdata, handles)
% hObject    handle to edit16 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit16 as text
%        str2double(get(hObject,'String')) returns contents of edit16 as a double


% --- Executes during object creation, after setting all properties.
function edit16_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit16 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edit17_Callback(hObject, eventdata, handles)
% hObject    handle to edit17 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit17 as text
%        str2double(get(hObject,'String')) returns contents of edit17 as a double


% --- Executes during object creation, after setting all properties.
function edit17_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit17 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edit18_Callback(hObject, eventdata, handles)
% hObject    handle to edit18 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit18 as text
%        str2double(get(hObject,'String')) returns contents of edit18 as a double
mask_size = str2num(get(hObject,'String'));
if(mask_size < 3) errordlg('Error, Mask Size>= 3.');
end
tmp = mod(mask_size,2);
if(tmp == 0) 
    errordlg('Error,mask for estimator needs to have uneven size (eg. 5x5).');
end
handles.edit18 = mask_size;
guidata(hObject,handles)


% --- Executes during object creation, after setting all properties.
function edit18_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit18 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in pushbutton24.
function pushbutton24_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton24 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edit19_Callback(hObject, eventdata, handles)
% hObject    handle to edit19 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit19 as text
%        str2double(get(hObject,'String')) returns contents of edit19 as a double


% --- Executes during object creation, after setting all properties.
function edit19_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit19 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in pushbutton25.
function pushbutton25_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton25 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edit20_Callback(hObject, eventdata, handles)
% hObject    handle to edit20 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit20 as text
%        str2double(get(hObject,'String')) returns contents of edit20 as a double


% --- Executes during object creation, after setting all properties.
function edit20_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit20 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edit21_Callback(hObject, eventdata, handles)
% hObject    handle to edit21 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit21 as text
%        str2double(get(hObject,'String')) returns contents of edit21 as a double
n_tests = str2num(get(hObject,'String'));
if(n_tests <= 0.0) errordlg('Error, Number tests > 0.');
end

handles.edit21 = n_tests;
guidata(hObject,handles)


% --- Executes during object creation, after setting all properties.
function edit21_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit21 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edit22_Callback(hObject, eventdata, handles)
% hObject    handle to edit22 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit22 as text
%        str2double(get(hObject,'String')) returns contents of edit22 as a double
ENL_tol = str2num(get(hObject,'String'));
if(ENL_tol < 0.0) errordlg('Error, ENL > 0.');
end

handles.edit22 = ENL_tol;
guidata(hObject,handles)


% --- Executes during object creation, after setting all properties.
function edit22_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit22 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edit23_Callback(hObject, eventdata, handles)
% hObject    handle to edit23 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit23 as text
%        str2double(get(hObject,'String')) returns contents of edit23 as a double


% --- Executes during object creation, after setting all properties.
function edit23_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit23 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edit24_Callback(hObject, eventdata, handles)
% hObject    handle to edit24 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit24 as text
%        str2double(get(hObject,'String')) returns contents of edit24 as a double
Mask_size_homogeneous = str2num(get(hObject,'String'));
if(Mask_size_homogeneous < 3) errordlg('Error, Mask Size >= 3.');
end
tmp = mod(Mask_size_homogeneous,2);
if(tmp == 0) 
    errordlg('Error,mask for estimator needs to have uneven size (eg. 5x5).');
end

handles.edit24 = Mask_size_homogeneous;
guidata(hObject,handles)

% --- Executes during object creation, after setting all properties.
function edit24_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit24 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edit25_Callback(hObject, eventdata, handles)
% hObject    handle to edit25 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit25 as text
%        str2double(get(hObject,'String')) returns contents of edit25 as a double


% --- Executes during object creation, after setting all properties.
function edit25_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit25 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edit26_Callback(hObject, eventdata, handles)
% hObject    handle to edit26 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit26 as text
%        str2double(get(hObject,'String')) returns contents of edit26 as a double


% --- Executes during object creation, after setting all properties.
function edit26_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit26 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edit27_Callback(hObject, eventdata, handles)
% hObject    handle to edit27 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit27 as text
%        str2double(get(hObject,'String')) returns contents of edit27 as a double
number_levels = str2num(get(hObject,'String'));
if(number_levels <= 0.0) errordlg('Error, Number of levels >= 0.');
end
% if(mod(number_levels,8) > 0) errordlg('Error, Number of levels must be power of 8.');
% end

handles.edit27 = number_levels;
guidata(hObject,handles)

% --- Executes during object creation, after setting all properties.
function edit27_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit27 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edit28_Callback(hObject, eventdata, handles)
% hObject    handle to edit28 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit28 as text
%        str2double(get(hObject,'String')) returns contents of edit28 as a double
ponderation = str2num(get(hObject,'String'));
if(ponderation < 0.0) errordlg('Error, Ponderation (w) >= 0.');
end
if(ponderation > 1.0) errordlg('Error, Ponderation (w) <= 1.');
end

handles.edit28 = ponderation;
guidata(hObject,handles)

% --- Executes during object creation, after setting all properties.
function edit28_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit28 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in pushbutton29.
function pushbutton29_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton29 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Estimators for H0
global ENL
tmp = '';set(handles.edit9,'String',tmp);
tmp = '';set(handles.edit10,'String',tmp);
tmp = '';set(handles.edit19,'String',tmp);
tmp = '';set(handles.edit20,'String',tmp);
tmp = '';set(handles.edit29,'String',tmp);
tmp = '';set(handles.edit30,'String',tmp);
clc

input_image = handles.pushbutton9;
H0 = input_image.H0;
[m n] = size(H0);

 option_intensity = handles.radiobutton5;
 option_amplitude = handles.radiobutton6;

if(option_intensity == 1) %intensity mode
   option = 0;
    display('Intensity Mode');
elseif(option_amplitude == 1) %amplitude mode
    option = 1;
     display('Amplitude Mode');
end

%Haralick's homogeneity
offset_tmp= handles.edit6;
levels_Haralick = handles.edit27;
% All directions
offset = [0 offset_tmp; -offset_tmp offset_tmp; -offset_tmp 0; -offset_tmp -offset_tmp];
glcm_data = graycomatrix(mat2gray(H0),'NumLevels',levels_Haralick, 'Offset', offset,'Symmetric',true);

% The average
tmp = glcm_data(:,:,1) + glcm_data(:,:,2) + glcm_data(:,:,3) + glcm_data(:,:,4);
glcm_average = tmp / 4;

% Haralick's descriptors for ratio image
Haralicks = GLCMFeatures(glcm_average);

% homogeneity 
homogeneity = Haralicks.homogeneity;
set(handles.edit10,'String',homogeneity);


%homogeneity for ratio disorderered
n_random_tests = handles.edit21;
variation = zeros(n_random_tests,1);
homogeneity_disordered = zeros(n_random_tests,1);
mn = m*n;
II = H0(:);

for ii = 1: n_random_tests
    r = randperm(mn);
    I_random = zeros(mn,1);
    for i = 1:numel(r)
        I_random(i) = II(r(i));
    end
    %make the matrix
    H0_tmp = reshape(I_random,m,n);
    % All directions
    glcm_data = graycomatrix(mat2gray(H0_tmp),'NumLevels',levels_Haralick, 'Offset', offset,'Symmetric',true);
   
    % The average
    tmp = glcm_data(:,:,1) + glcm_data(:,:,2) + glcm_data(:,:,3) +glcm_data(:,:,4);
    glcm_average = tmp / 4;
    
    % Haralick's descriptors for ratio image
    Haralicks = GLCMFeatures(glcm_average);
    
    % homogeneity
    homogeneity_disordered(ii) = Haralicks.homogeneity
  end

homogeneity_disordered = mean(homogeneity_disordered);
set(handles.edit19,'String',homogeneity_disordered);
delta_homogeneity = 100 * abs((homogeneity - homogeneity_disordered)/homogeneity);
set(handles.edit20,'String',delta_homogeneity);

% ENL_M EAN estimator (residual)
% ENL Mean measured on Ratio but within homogeneous areas within the Noisy
% image
ENL = handles.edit1;
mask_ENL = handles.edit24;
ENL_tol = handles.edit22;
ponderation = handles.edit28;

if(option ==0) H0_ENL = blkproc(H0,[mask_ENL mask_ENL],@estimator_ENL_intensity);
else H0_ENL = blkproc(H0,[mask_ENL mask_ENL],@estimator_ENL_amplitude);
end
[mm nn] = size(H0_ENL);
H0_ENL = H0_ENL(1:mm-1,1:nn-1);
[m n] = size(H0_ENL);
%look for homogeneous areas
inf_tol = ENL - ENL * ENL_tol / 100;
sup_tol = ENL + ENL * ENL_tol / 100;
H0_homogeneous =(H0_ENL > inf_tol) & (H0_ENL <  sup_tol);


if(option ==0) H0_ENL = blkproc(H0,[mask_ENL mask_ENL],@estimator_ENL_intensity);
else H0_ENL = blkproc(H0,[mask_ENL mask_ENL],@estimator_ENL_amplitude);
end
[mm nn] = size(H0_ENL);
H0_ENL = H0_ENL(1:mm-1,1:nn-1);

Mean_H0 = blkproc(H0, [mask_ENL mask_ENL], @Mean_image);
[mm nn] = size(Mean_H0);
Mean_H0 = Mean_H0(1:mm-1,1:nn-1);

TOTAL_ENL_Mean_H0 = 0.0;
r_ENL = 0;
r_mu = 0;
for i = 1: m
    for j = 1: n
        if(H0_homogeneous(i,j) == 1)
            res_ENL = abs(ENL - H0_ENL(i,j))/ENL;
            res_mu =  abs(1-Mean_H0(i,j));
            TOTAL_ENL_Mean_H0 = TOTAL_ENL_Mean_H0 + (res_ENL + res_mu) * 100/2;
        end
    end
end

total_homogeneous_area = numel(find(H0_homogeneous > 0));
TOTAL_ENL_Mean_H0 = TOTAL_ENL_Mean_H0/total_homogeneous_area;
set(handles.edit9,'String',TOTAL_ENL_Mean_H0);
set(handles.edit30,'String',total_homogeneous_area);
k = handles.edit48; %coefficiento to scale delta_h
set(handles.edit29,'String',(TOTAL_ENL_Mean_H0 * ponderation  + k * delta_homogeneity * (1 - ponderation)));

ENL_Mean_Data = struct('Homogeneity', homogeneity,'Homogeneity_disordered', homogeneity_disordered, 'Variation',delta_homogeneity, 'ENL_MEAN', TOTAL_ENL_Mean_H0 );
handles.pushbutton10 = ENL_Mean_Data;
guidata(hObject,handles)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edit29_Callback(hObject, eventdata, handles)
% hObject    handle to edit29 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit29 as text
%        str2double(get(hObject,'String')) returns contents of edit29 as a double


% --- Executes during object creation, after setting all properties.
function edit29_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit29 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edit30_Callback(hObject, eventdata, handles)
% hObject    handle to edit30 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit30 as text
%        str2double(get(hObject,'String')) returns contents of edit30 as a double


% --- Executes during object creation, after setting all properties.
function edit30_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit30 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in checkbox1.
function checkbox1_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox1
mode = get(hObject,'Value');
handles.checkbox1 = 1;
guidata(hObject,handles)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in checkbox2.
function checkbox2_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox2
mode = get(hObject,'Value');
handles.checkbox2 = 1;
guidata(hObject,handles)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in checkbox3.
function checkbox3_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox3
mode = get(hObject,'Value');
handles.checkbox3 = 1;
guidata(hObject,handles)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in checkbox4.
function checkbox4_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox4
mode = get(hObject,'Value');
handles.checkbox4 = 1;
guidata(hObject,handles)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in checkbox6.
function checkbox6_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox6
mode = get(hObject,'Value');
handles.checkbox6 = 1;
guidata(hObject,handles)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in checkbox7.
function checkbox7_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox7
mode = get(hObject,'Value');
handles.checkbox7 = 1;
guidata(hObject,handles)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in checkbox8.
function checkbox8_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox8
mode = get(hObject,'Value');
handles.checkbox8 = 1;
guidata(hObject,handles)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in checkbox9.
function checkbox9_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox9
mode = get(hObject,'Value');
handles.checkbox9 = 1;
guidata(hObject,handles)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edit37_Callback(hObject, eventdata, handles)
% hObject    handle to edit37 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit37 as text
%        str2double(get(hObject,'String')) returns contents of edit37 as a double


% --- Executes during object creation, after setting all properties.
function edit37_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit37 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edit40_Callback(hObject, eventdata, handles)
% hObject    handle to edit40 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit40 as text
%        str2double(get(hObject,'String')) returns contents of edit40 as a double


% --- Executes during object creation, after setting all properties.
function edit40_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit40 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edit41_Callback(hObject, eventdata, handles)
% hObject    handle to edit41 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit41 as text
%        str2double(get(hObject,'String')) returns contents of edit41 as a double


% --- Executes during object creation, after setting all properties.
function edit41_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit41 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in pushbutton30.
function pushbutton30_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton30 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%OPtimization of FANS Filter
global ENL

clc
tmp = '';set(handles.edit37,'String',tmp);
tmp = '';set(handles.edit40,'String',tmp);
tmp = '';set(handles.edit41,'String',tmp);

 mode_1 = handles.radiobutton5;  % intensity
 mode_2 = handles.radiobutton6;  % amplitude

% mode_1 = 0;
% mode_2 = 1;

if(mode_1 == 1) %intensity data
    option = 1;
elseif mode_2 == 1; % amplitude data
    option = 0;
end


input_images = handles.popupmenu1;
I = input_images.Noisy_image;
[m n] = size(I);

mask_ENL = handles.edit24;
ENL_tol = handles.edit22;
ponderation = handles.edit28;
% option_intensity = handles.radiobutton1;
% option_amplitude = handles.radiobutton2;

option_intensity = 0;
option_amplitude = 1;

if(option_intensity == 1) %intensity mode
    option = 0;
    display('Intensity Mode');
elseif(option_amplitude == 1) %amplitude mode
    option = 1;
    display('Amplitude Mode');
end

%Haralick's homogeneity
offset_tmp= handles.edit6;
levels_Haralick = handles.edit27;
%%%%%%%%%%%%%%%%%%Haralick's homogeneity by using Matlab functions%%%%%%
% % All directions
offset = [0 offset_tmp; -offset_tmp offset_tmp; -offset_tmp 0; -offset_tmp -offset_tmp];
%%%%%%%%%%%%%%%%%%Haralick's homogeneity by using Matlab functions%%%%%%

% Optimization
% Get input parameters
min_rows_cols = handles.edit31;
max_rows_cols = handles.edit32;
if(max_rows_cols  <= min_rows_cols)
    errordlg('Input parameters are wrong Max Rows&Cols  > MIn Rows&Cols.');
end
step_rows_cols = handles.edit33;

min_false_alarm = handles.edit34;
max_false_alarm = handles.edit35;
if(max_false_alarm <= min_false_alarm)
    errordlg('Input parameters are wrong Max False Alarm Probability  > Min False Alarm Probability.');
end
step_false_alarm = handles.edit36;

% Wavelet window decimation
window = zeros(1,8);
window(1) = handles.checkbox1;
window(2) = handles.checkbox2;
window(3) = handles.checkbox3;
window(4) = handles.checkbox4;
window(5) = handles.checkbox6;
window(6) = handles.checkbox7;
window(7) = handles.checkbox8;
window(8) = handles.checkbox9;

inf_tol = ENL - ENL * ENL_tol / 100;
sup_tol = ENL + ENL * ENL_tol / 100;
n_random_tests = handles.edit21;
%Exhaustive seach
M_estimator = 1e100;
for r = min_rows_cols: step_rows_cols: max_rows_cols
    display('rows/cols')
    r
    for s = min_false_alarm: step_false_alarm: max_false_alarm
        display('False Alarm')
        s
        for t = 1: 8
            if(window(t) == 1)
              display('Wavelet transform');
              t
                parameters = struct('param1',r, 'param2',s, 'param3',t);
                if(option == 1)  %intensity
                  F = FANS_OPTIMO(sqrt(I),ENL,parameters);
                  F = F.^2;
                elseif(option == 0) %amplitude
                    F = FANS_OPTIMO(I,ENL,parameters);
                end
                ratio = I ./F;               
                glcm_data = graycomatrix(mat2gray(ratio),'NumLevels',levels_Haralick, 'Offset', offset,'Symmetric',true,'GrayLimits',[]);
                % The average
                tmp = glcm_data(:,:,1) + glcm_data(:,:,2) + glcm_data(:,:,3) + glcm_data(:,:,4);
                glcm_average = tmp / 4;
                % Haralick's descriptors for ratio image
                Haralicks = GLCMFeatures(glcm_average);
                % homogeneity
                homogeneity = Haralicks.homogeneity;                
                %Ratio Disordered
                %homogeneity for ratio disorderered
                homogeneity_disordered = zeros(n_random_tests,1);
                [m n] = size(I);
                mn = m*n;
                II = ratio(:);
                for ii = 1: n_random_tests
                    rr = randperm(mn);
                    I_random = zeros(mn,1);
                    for i = 1:numel(rr)
                        I_random(i) = II(rr(i));
                    end
                    ratio_tmp = zeros(m,n);
                    %make the matrix
                    count = 1;
                    for i = 1: m
                        for j = 1:n
                            ratio_tmp(i,j) = I_random(count);
                            count = count + 1;
                        end
                    end
                    %%%%%%%%Haralick's homogeneity by using Matlab functions%%%%%
                    % All directions
                    glcm_data = graycomatrix(mat2gray(ratio_tmp),'NumLevels',levels_Haralick, 'Offset', offset,'Symmetric',true,'GrayLimits',[]);
                    % The average
                    tmp = glcm_data(:,:,1) + glcm_data(:,:,2) + glcm_data(:,:,3) +glcm_data(:,:,4);
                    glcm_average = tmp / 4;
                    % Haralick's descriptors for ratio image
                    Haralicks = GLCMFeatures(glcm_average);                    
                    % homogeneity
                    homogeneity_disordered(ii) = Haralicks.homogeneity;
                    %%%%%%%%Haralick's homogeneity by using Matlab functions%%%%%
                end
                homogeneity_disordered = mean(homogeneity_disordered);
                delta_homogeneity = 100 * abs((homogeneity - homogeneity_disordered)/homogeneity);
                  
                %ENL_M EAN estimator (residual)
                % ENL Mean measured on Ratio but within homogeneous areas within the Noisy
                % image                
                
                if(option ==0) Noisy_ENL = blkproc(I,[mask_ENL mask_ENL],@estimator_ENL_intensity);
                else Noisy_ENL = blkproc(I,[mask_ENL mask_ENL],@estimator_ENL_amplitude);
                end
                [mm nn] = size(Noisy_ENL);
                Noisy_ENL = Noisy_ENL(1:mm-1,1:nn-1);
                [m n] = size(Noisy_ENL);
                %look for homogeneous areas
                Noisy_homogeneous =(Noisy_ENL > inf_tol) & (Noisy_ENL <  sup_tol);
                               
                if(option ==0) Ratio_ENL = blkproc(ratio,[mask_ENL mask_ENL],@estimator_ENL_intensity);
                else Ratio_ENL = blkproc(ratio,[mask_ENL mask_ENL],@estimator_ENL_amplitude);
                end
                [mm nn] = size(Ratio_ENL);
                Ratio_ENL = Ratio_ENL(1:mm-1,1:nn-1);
                
                Mean_Ratio = blkproc(ratio, [mask_ENL mask_ENL], @Mean_image);
                [mm nn] = size(Mean_Ratio);
                Mean_Ratio = Mean_Ratio(1:mm-1,1:nn-1);
                
                TOTAL_ENL_Mean_ratio = 0.0;
                for i = 1: m
                    for j = 1: n
                        if(Noisy_homogeneous(i,j) == 1)
                            TOTAL_ENL_Mean_ratio = TOTAL_ENL_Mean_ratio + (abs(Noisy_ENL(i,j) - Ratio_ENL(i,j)) + abs(1-Mean_Ratio(i,j)))/2;
                        end
                    end
                end
                
                total_homogeneous_area = numel(find(Noisy_homogeneous > 0));
                TOTAL_ENL_Mean_ratio = TOTAL_ENL_Mean_ratio/total_homogeneous_area;
                M = TOTAL_ENL_Mean_ratio * ponderation  + delta_homogeneity * (1 - ponderation);   
                if(M < M_estimator)
                    M_estimator = M
                    optimal_design = [r,s,t]
                end
           end
        end                
    end
end
display('FANS optimal design:');
optimal_design
set(handles.edit37, 'String',delta_homogeneity);
set(handles.edit40, 'String', TOTAL_ENL_Mean_ratio);
set(handles.edit41, 'String', M_estimator);
M_Estimator = struct('parameters', optimal_design);
handles.pushbutton30 = M_Estimator;
guidata(hObject,handles)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edit34_Callback(hObject, eventdata, handles)
% hObject    handle to edit34 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit34 as text
%        str2double(get(hObject,'String')) returns contents of edit34 as a double
min_false_alarm = str2num(get(hObject,'String'));
if(min_false_alarm <= 0) errordlg('Error, Min. false alarm probability > 0.');
end

handles.edit34 = min_false_alarm;
guidata(hObject,handles)

% --- Executes during object creation, after setting all properties.
function edit34_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit34 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edit35_Callback(hObject, eventdata, handles)
% hObject    handle to edit35 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit35 as text
%        str2double(get(hObject,'String')) returns contents of edit35 as a double
max_false_alarm = str2num(get(hObject,'String'));
if(max_false_alarm <= 0) errordlg('Error, Max. false alarm probability > 0.');
end

handles.edit35 = max_false_alarm;
guidata(hObject,handles)

% --- Executes during object creation, after setting all properties.
function edit35_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit35 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edit36_Callback(hObject, eventdata, handles)
% hObject    handle to edit36 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit36 as text
%        str2double(get(hObject,'String')) returns contents of edit36 as a double
step = str2num(get(hObject,'String'));
if(step <= 0) errordlg('Error, Step > 0.');
end

handles.edit36 = step;
guidata(hObject,handles)

% --- Executes during object creation, after setting all properties.
function edit36_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit36 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edit31_Callback(hObject, eventdata, handles)
% hObject    handle to edit31 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit31 as text
%        str2double(get(hObject,'String')) returns contents of edit31 as a double

min_rows_cols = str2num(get(hObject,'String'));
if(min_rows_cols < 4) errordlg('Error, min_rows_cols >= 4.');
end

handles.edit31 = min_rows_cols;
guidata(hObject,handles)

% --- Executes during object creation, after setting all properties.
function edit31_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit31 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edit32_Callback(hObject, eventdata, handles)
% hObject    handle to edit32 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit32 as text
%        str2double(get(hObject,'String')) returns contents of edit32 as a double
max_rows_cols = str2num(get(hObject,'String'));

handles.edit32 = max_rows_cols;
guidata(hObject,handles)


% --- Executes during object creation, after setting all properties.
function edit32_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit32 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function edit33_Callback(hObject, eventdata, handles)
% hObject    handle to edit33 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit33 as text
%        str2double(get(hObject,'String')) returns contents of edit33 as a double

step= str2num(get(hObject,'String'));
if(step <= 0) errordlg('Error, step > 0.');
end
handles.edit33 = step;
guidata(hObject,handles)

% --- Executes during object creation, after setting all properties.
function edit33_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit33 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in pushbutton31.
function pushbutton31_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton31 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Evaluates FANS filter in the optimal solution obtained

% Gets the optimal solution
param = handles.pushbutton30;
parameters = struct('param1',param.parameters(1), 'param2',param.parameters(2), 'param3',param.parameters(3));
% Gets the Noisy image to filter
input_images = handles.popupmenu1;
I = input_images.Noisy_image;

 mode_1 = handles.radiobutton5;  % intensity
 mode_2 = handles.radiobutton6;  % amplitude

% mode_1 = 0;
% mode_2 = 1;

if(mode_1 == 1) %intensity data
    option = 1;
elseif mode_2 == 1; % amplitude data
    option = 0;
end

global ENL
%Run FANS filter (FANS works in amplitude mode)
display('Running FANS with optimal design');
if(option == 1) % intensity
    F = FANS_OPTIMO(sqrt(I),ENL,parameters);
    F = F.^2;
elseif (option == 0) % amplitude
    F = FANS_OPTIMO(I,ENL,parameters);
end
ratio = I ./F;   

% Plot filtered image
axes(handles.axes9);
escala_display=mean(F(:))*3;imshow(imresize(F,1),[0,escala_display]);
% Plot ratio image
axes(handles.axes10);
escala_display=mean(ratio(:))*3; imshow(imresize(ratio,1),[0,escala_display]);
display('Finished FANS with optimal design: Filtered and Ratio images updated');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%--- Executes on button press in radiobutton5.
function radiobutton5_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% 
% Hint: get(hObject,'Value') 
% returns toggle state of radiobutton5

option = get(hObject,'Value');
handles.radiobutton5 = 1;
handles.radiobutton6 = 0;
guidata(hObject,handles)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--- Executes on button press in radiobutton6.
function radiobutton6_Callback(hObject, eventdata, handles)
% hObject    handle to radiobutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% 
% Hint: get(hObject,'Value') returns toggle state of radiobutton6
option = get(hObject,'Value');
handles.radiobutton6 = 1;
handles.radiobutton5 = 0;
guidata(hObject,handles)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
function edit42_Callback(hObject, eventdata, handles)
% hObject    handle to edit42 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit42 as text
%        str2double(get(hObject,'String')) returns contents of edit42 as a double


% --- Executes during object creation, after setting all properties.
function edit42_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit42 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit43_Callback(hObject, eventdata, handles)
% hObject    handle to edit43 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit43 as text
%        str2double(get(hObject,'String')) returns contents of edit43 as a double


% --- Executes during object creation, after setting all properties.
function edit43_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit43 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit44_Callback(hObject, eventdata, handles)
% hObject    handle to edit44 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit44 as text
%        str2double(get(hObject,'String')) returns contents of edit44 as a double


% --- Executes during object creation, after setting all properties.
function edit44_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit44 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function edit45_Callback(hObject, eventdata, handles)
% hObject    handle to edit45 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit45 as text
%        str2double(get(hObject,'String')) returns contents of edit45 as a double


% --- Executes during object creation, after setting all properties.
function edit45_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit45 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function edit46_Callback(hObject, eventdata, handles)
% hObject    handle to edit46 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit46 as text
%        str2double(get(hObject,'String')) returns contents of edit46 as a double


% --- Executes during object creation, after setting all properties.
function edit46_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit46 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function edit47_Callback(hObject, eventdata, handles)
% hObject    handle to edit47 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit47 as text
%        str2double(get(hObject,'String')) returns contents of edit47 as a double


% --- Executes during object creation, after setting all properties.
function edit47_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit47 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  M Estimator (homogeneity, ENL, Mean)
%  Reproducible research for article: 
%  L. Gomez, R. Ospina, A. C. Frery
%  "Unassisted quantitative evaluation of despeckling filters"
%  Work by  L. Gomez, R. Ospina, A. C. Frery; codified to Matlab by L. Gomez 
%  Run well on Matlab 2014a
%  February 2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Please use it freely and cite the reference paper mentioned above.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function edit48_Callback(hObject, eventdata, handles)
% hObject    handle to edit48 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit48 as text
%        str2double(get(hObject,'String')) returns contents of edit48 as a double
k = str2num(get(hObject,'String'));
if(k < 0.0) errordlg('Error, k coefficient (k) >= 0.');
end

handles.edit48 = k;
guidata(hObject,handles)


% --- Executes during object creation, after setting all properties.
function edit48_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit48 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
