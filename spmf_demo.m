% Code for Sparse Probabilistic Matrix Factorization for Recommendation
% System
%
% Auther: Wang Peng
% Beijing Jiaotong University
% Date: Dec 30, 2014
clear;
clc;
global datasetname;
datasetname = 'netflix';
addpath('PCP');
addpath('Utilities');
addpath('mex');
rand('state',0);
randn('state',0);

%% load data and initialise the parameters
if strcmp(datasetname, 'movielens')
    load moviedata;
else if strcmp(datasetname, 'netflix')
        load netflix_data;
    end
end
opts.debug = 1;
opts.r =50 ;                               % the number of latent factor
opts.maxIter = 20  ;                       % the max gibbs sample iterator number
opts.burnin = 10 ;                         % the burn in gibbs sampling number
opts.mean_rating = mean(train_vec(:,3));   % the mean of train_vec
opts.p = 0    ;                            % hyperparameter p of GIG as we don't use p = -1/2 
opts.a = 10;                             % hyperparameter a of GIG
%opts.a = 1 / (opts.r * 1000);
opts.b = 1e0;                              % hyperparameter b of GIG
opts.b = opts.r;
opts.b = 50;
opts.k = 0;                                % hyperparameter k of inverse gamma
opts.theta = 0;                            % hyperparameter theat of inverse gamma
opts.delta = 0.65;                         % for debug
opts.probe_vec = probe_vec;                % probe vec to calculate the RMSE 
opts.train_vec = train_vec;                % train vec to calculate the RMSE
probe_num = length(probe_vec);             % the length of probe vector
mean_rating = mean(train_vec(:,3));        % the mean of probe_vec
pairs_pr = length(probe_vec);              % the number of probe ratings
ratings_test = double(probe_vec(:,3));     % the probe rating
makematrix;
R = count;
clear count;

filename = strcat(datasetname, '_pmf_weight_');
filename = strcat(filename, num2str(opts.r));
load(filename);
U = w1_P1;
V = w1_M1;
clear w1_P1 w1_M1;

%% gibbs sampling
[U, V, train_err, probe_err] = SPMF(R, opts, U, V);

%% calculate the RMSE
probe_rat = pred(V, U, probe_vec, opts.mean_rating);
temp = (ratings_test - probe_rat).^2;
err = sqrt( sum(temp)/pairs_pr);
fprintf('the error of this experienment is %f\n', err);