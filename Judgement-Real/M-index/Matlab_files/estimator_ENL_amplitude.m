function [ENL_estimate]= estimator_ENL_amplitude(x)
% amplitude mode

mean_ROI= mean2(x);
std_ROI = std2(x);
ENL_estimate =   0.2732 * (mean_ROI^2 / std_ROI^2);

