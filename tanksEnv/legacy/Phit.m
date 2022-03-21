clc;clear;close all;

vis = 10;
R50 = 10;

r = linspace(0,1.5*vis)';
sig = 1.0./ (1+exp((r-R50).*12./R50));

plot(r,100*sig)
ylim([0 100])