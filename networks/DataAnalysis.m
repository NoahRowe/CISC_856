close all;
clc;
clear all;

%% Initialization
%fileID = fopen('20211022_10000ConPreplaced.txt');
%fileID = fopen('20211025_10000ConPreplaced05AplphaShifted.txt'); % Good Run!
fileID = fopen('20220330_RNNTraining.txt');
data = fscanf(fileID,'%f');
n = 300; %Number of Episodes for the run
%maxr = 10; %maximum reward

%% Raw Data
episodes = 1:n;

% plot(episodes,data)
% xlabel('Episode')
% ylabel('Reward')
% hold on;

%% Smoothed Data
sdata = smoothdata(data,'gaussian',15);
sum = 0;

plot(episodes,data)
hold on;
plot(episodes,sdata,'linewidth',2)
%yline(maxr)
%ylim([-10 12]);
xlabel('Episode','FontSize',14)
ylabel('Reward','FontSize',14)
legend('Reward per episode','15 Point Moving Average','Max Reward','Location','best')
hold off;

