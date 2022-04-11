function [ENL_estimate]= estimator_ENL_intensity(x)
% intensity mode

mean_ROI= mean2(x);
std_ROI = std2(x);
ENL_estimate =  mean_ROI^2 / std_ROI^2;

