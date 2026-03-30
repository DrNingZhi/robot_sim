clear
clc

addpath(genpath(pwd));

pref=[zeros(1,2001),ones(1,1000)*0.3,...
      ones(1,500)*0.6,ones(1,500)*0.9,...
      ones(1,1000)*1.2,ones(1,1000)*1.5];
% pref=[zeros(1,2001),ones(1,1000)*0.3,...
%        zeros(1,1000),ones(1,1000)*0.3,zeros(1,10000)];
dt=0.001;
zc=1;
subplot(1,2,1)
gait=fun_mpc_gait(pref,zc,dt);