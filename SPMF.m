function [U V overall_train_err, overall_probe_err] = SPMF(R, opts, U, V)
    global datasetname;
    rand('state',0);
    randn('state',0);
    
    [m, n] = size(R);
    % if there are no init U and V, we rand U and V.
    if nargin < 3
        U = randn(m, opts.r);
        V = randn(n, opts.r);
    end
    % the gibbs sampling series
    accU = zeros(m, opts.r);
    accV = zeros(n, opts.r);
    % initial the hyperparameters
    eta_U = ones(m, opts.r);
    eta_V = ones(n, opts.r);
    tau_U = ones(m, opts.r);
    tau_V = ones(n, opts.r);
    list = find(R > 0);
    probe_rat_all = pred(V, U, opts.probe_vec,opts.mean_rating);
    overall_train_err = zeros(opts.maxIter, 1);
    overall_probe_err = zeros(opts.maxIter, 1);
    filename = 'pmf_weight_';
    filename = strcat(filename, num2str(opts.r));
    load(filename);
    
    for iter = 1 : opts.maxIter
        if (8 == iter)
            fprintf('debug');
        end
        fprintf('The iter is %d \n', iter);
       %% Sample the hyperparameters including eta tau and delta
        for i = 1 : m
            % Sample eta and tau in inverse Gaussian
            res = abs(U(i, :));
            tau_U(i, :) = drawFromIG(sqrt(eta_U(i,:)) ./ res, eta_U(i,:));
            tau_U(i, :) = 1 ./ tau_U(i, :);
            tau_U(i, :) = tau_U(i, :) + 1e-6;
            if opts.debug
                eta_U(i, :) = drawFromGIG(opts.p + 1, opts.b, opts.a + tau_U(i, :));
            else
                eta_U(i, :) = drawFromIG(sqrt((tau_U(i, :) + opts.a) ./ opts.b), tau_U(i, :) + opts.a);
            end
            eta_U(i, :) = 1 ./ eta_U(i, :); 
        end
        
        for i = 1 : n
            % Sample eta and tau in inverse Gaussian
            res = abs(V(i, :));
            tau_V(i, :) = drawFromIG(sqrt(eta_V(i,:)) ./ res, eta_V(i,:));
            tau_V(i, :) = 1 ./ tau_V(i, :);
            tau_V(i, :) = tau_V(i, :) + 1e-6;
            if opts.debug
            	eta_V(i, :) = drawFromGIG(opts.p + 1, opts.b, opts.a + tau_V(i, :));
            else
                eta_V(i, :) = drawFromIG(sqrt((tau_V(i, :) + opts.a) ./ opts.b), tau_V(i, :) + opts.a);
            end
            eta_V(i, :) = 1 ./ eta_V(i, :); 
        end 
        
        UVt = U * V' + opts.mean_rating;
        ff = find(UVt > 5); UVt(ff)=5;
        ff = find(UVt < 1); UVt(ff)=1;
        square_err = sum((R(list) - UVt(list)) .^ 2);
        % sample the delta
	    delta = 1/gamrnd((m * n)/500+1+opts.k, 1/(square_err / 2 + opts.theta));
	    delta = 1/gamrnd((m * n)*square_err*0.4/(1e7)+1+opts.k, 1/(square_err / 2 + opts.theta));
        delta = 1/gamrnd((m * n)/2+1+opts.k, 1/(square_err / 2 + opts.theta));
        delta = opts.delta;
        
       %% Sample the parameters U and V
        % sample U
        for i = 1 : m
            rating_list = find(R(i, :) > 0);
            rating_u = R(i, rating_list) - opts.mean_rating;
            Lambda_u =diag(1 ./ tau_U(i, :)) + V(rating_list, :)' * V(rating_list, :) ./ delta;
            covar = inv(Lambda_u);
            covar = (covar + covar') / 2;
            mu_u  = covar * (V(rating_list, :)' * rating_u' ./ delta);
            if length(find(eig(covar) <= 0) ~= 0)
                fprintf('the covar matrix is not positive\n');
            end
            lam = chol(covar); lam=lam'; 
            U(i,:) = lam*randn(opts.r,1)+mu_u;
        end
        
        % sample V
        for i = 1 : n
            rating_list = find(R(:, i) > 0);
            rating_v = R(rating_list, i) - opts.mean_rating;
            Lambda_v = diag(1 ./ tau_V(i, :)) + U(rating_list, :)' * U(rating_list, :) ./ delta;
            covar = inv(Lambda_v);
            covar = (covar + covar') / 2;
            mu_v  = covar * (U(rating_list, :)' * rating_v ./ delta);
            if length(find(eig(covar) <= 0) ~= 0)
                fprintf('the covar matrix is not positive\n');
            end
            lam = chol(covar); lam=lam'; 
            V(i,:) = lam*randn(opts.r,1)+mu_v;
        end
        
        probe_rat = pred(V, U, opts.probe_vec,opts.mean_rating);
        if iter > opts.burnin
            accU = accU + U;
            accV = accV + V;
            probe_rat_all = probe_rat_all + probe_rat;
        end
        
        % for debug we show the RMSE of every epoch
        train_rat = zeros(length(opts.train_vec),1);
        if strcmp(datasetname, 'movielens')
            for i = 1 : 9
                train_rat(((i - 1) * 100000 + 1):(100000 * i)) = sum(V(opts.train_vec(((i - 1) * 100000 + 1):(100000 * i),2),:).* ...
                    U(opts.train_vec(((i - 1) * 100000 + 1):(100000 * i),1),:),2) + opts.mean_rating;
            end
            ff = find(train_rat>5); train_rat(ff)=5;
            ff = find(train_rat<1); train_rat(ff)=1;
        else if strcmp(datasetname, 'netflix')
                 train_rat = pred(V,U,opts.train_vec,opts.mean_rating);
            end
        end
        
        temp = (double(opts.probe_vec(:, 3)) - probe_rat).^2;
        probe_err = sqrt(sum(temp)/length(opts.probe_vec));
        temp = (double(opts.train_vec(:, 3)) - train_rat).^2;
        train_err = sqrt(sum(temp)/length(opts.train_vec));
        overall_train_err(iter) = train_err;
        overall_probe_err(iter) = probe_err; 
        fprintf('the spmf RMSE is : train %f  probe %f\n', train_err, probe_err);
        
        % for debug we plot the fingure of PMF and SPMF
        result_pmf = zeros(10000,1);
        result_spmf = zeros(10000, 1);
        for i = 1 : n * opts.r
            if V(i) >= -5 & V(i) <= 5
                level = fix((V(i) + 5) / 0.001) + 1;
                result_spmf(level) = result_spmf(level) + 1;
            end
            if w1_M1(i) >= -5 & w1_M1(i) <= 5
                level = fix((w1_M1(i) + 5) / 0.001) + 1;
                result_pmf(level) = result_pmf(level) + 1;
            end            
        end
        fprintf('the kurtosis of pmf %d and spmf %d \n', max(result_pmf), max(result_spmf));
    end
    
    probe_rat_all = probe_rat_all / (opts.maxIter - opts.burnin + 1);
    temp = (double(opts.probe_vec(:, 3)) - probe_rat_all).^2;
    real_err = sqrt( sum(temp)/length(opts.probe_vec));
    fprintf('the real err of SPMF is %f \n', real_err);
    U = accU ./ (opts.maxIter - opts.burnin);
    V = accV ./ (opts.maxIter - opts.burnin);
    
    filename = strcat(datasetname, '_spmf_ret_');
    filename = strcat(filename, num2str(opts.r));
    filename = strcat(filename, '_');
    filename = strcat(filename, date);
    %save(filename);
    
    % plot the fingure of pdf
    plot(result_spmf, 'b');
    hold on;
    plot(result_pmf, 'r');
end
