clc;clear;close all;

vis = 10;
R50 = .8*vis;

r = linspace(0,vis)';
sig = 1.0./ (1+exp((r-R50).*12./R50));

plot(r,100*sig)
ylim([0 100])