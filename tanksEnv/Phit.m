clc;clear;close all;

r = linspace(0,20)';
R50 = 12;
sig = 1.0./ (1+exp((r-R50).*12./R50));

plot(r,100*sig)
ylim([0 100])