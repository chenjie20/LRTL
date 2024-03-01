close all;
clear;
clc;

addpath('data');
addpath('utility');

%---------------------- tips --------------------------------------------
% Required: Matlab 2019b or later. Matlab 2021b (recommended).
% Please run run_main_parameters.m to obtain proper parameters for 
% different datasets. The parameters given in this file may be not optimal 
% on different computers due to Pseudo random number generation.
%-------------------------------------------------------------------------

%---------------------- load data------------------------------------------
%---------------------data description-------------------------------------
%------nv represents the number of views-----------------------------------
%------Suppose the size of each cell is m * n, where m and n represent the 
% dimensionality and the number of the features, respectively. ------------
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% We may apply a PCA algorithm to preprocess the original features of all
%samples.
% The PCA algorithm will be skipped when the parameter is set to 0.
% -------------------------------------------------------------------------
new_dims = [0, 0, 0, 0];

% 1 MSRCv1; 2 reuters; 3 handwritten; 4 flower17; 5 proteinFold; 
%  6 COIL20; 7 100leaves; 8 Caltech101; 9 scene;
data_index = 1;
switch data_index
    case 1
        lambdas = [1, 1.5, 0.5, 1.2];
        betas = [2, 2, 5, 2];

        filename = "MSRCv1";
        load('MSRCv1.mat');
        n = length(Y);
        nv = size(X, 2);
        K = length(unique(Y));
        gnd = Y;
        data_views = cell(1, nv);
        for nv_idx = 1 : nv
             data_views{nv_idx} = X{nv_idx}';
        end
        
     case 2
        lambdas = [0.3, 0.2, 0.3, 0.3];
        betas = [0.5, 1, 0.5, 0.2];

        filename = "reuters";
        load('reuters.mat');
        K = size(category, 2);
        nv = size(X, 2);
        
        data_views = cell(1, nv);
        num_each_class = 100;
        total_num = num_each_class * K;
        n = total_num;
        gnd = zeros(1, total_num);
        for nv_idx = 1 : nv 
             dim = size(X{nv_idx}, 2);
             data_views{nv_idx} = zeros(dim, num_each_class * K);
        end
        rand('state', 100);
        for idx = 1 : K
           view_ids = find(Y == (idx - 1));
           len = length(view_ids);           
           rnd_idx = randperm(len);
           new_view_ids = view_ids(rnd_idx(1 :num_each_class));
           current_ids = ((idx - 1) * num_each_class + 1) : idx * num_each_class;
           gnd(1, current_ids) = idx;
           for nv_idx = 1 : nv 
               data_views{nv_idx}(:, current_ids) = X{nv_idx}(new_view_ids, :)';
           end
        end
        new_dims = [500, 300, 300, 200];

    case 3
        lambdas = [2, 3, 1.2, 1.2];
        betas = [1, 0.2, 5, 2];

        filename = "handwritten";
        load('handwritten.mat');
        n = length(Y);
        nv = size(X, 2);
        K = length(unique(Y));
        gnd = Y + 1;
        data_views = cell(1, nv);
        for nv_idx = 1 : nv
             data_views{nv_idx} = X{nv_idx}';
        end

    case 4
        lambdas = [5, 5, 1.2, 3];
        betas = [0.5, 2, 0.5, 2];

        filename = "flower17";
        load('flower17_Kmatrix.mat');
        n = length(Y);
        nv = size(KH, 3);
        K = length(unique(Y));
        gnd = Y;
        data_views = cell(1, nv);
        for nv_idx = 1 : nv
             data_views{nv_idx} = KH(:, :, nv_idx);
        end

     case 5
        lambdas = [0.4, 0.4, 0.6, 10];
        betas = [1, 1, 1, 0.5];

        filename = "proteinFold";
        load('proteinFold_Kmatrix.mat');
        n = length(Y);
        nv = size(KH, 3);
        K = length(unique(Y));
        gnd = Y;
        data_views = cell(1, nv);
        for nv_idx = 1 : nv
             data_views{nv_idx} = KH(:, :, nv_idx);
        end

    case 6
        lambdas = [0.1, 0.05, 0.1, 0.05];
        betas = [0.2, 0.5, 0.2, 0.1];

        filename = "COIL20";
        load('COIL20.mat');
        n = length(Y);
        nv = size(X, 2);
        K = length(unique(Y));
        gnd = Y;
        data_views = cell(1, nv);
        for nv_idx = 1 : nv
             data_views{nv_idx} = X{nv_idx}';
        end
        new_dims = [200, 200, 100, 100];

    case 7
        lambdas = [0.6, 0.5, 0.3, 0.4];
        betas = [2, 2, 2, 1];

        filename = "leaves";
        load('100leaves.mat');
        n = size(truelabel{1}, 1);
        nv = size(data, 2);
        K = length(unique(truelabel{1}));
        gnd = truelabel{1};
        data_views = cell(1, nv);
        for nv_idx = 1 : nv
             data_views{nv_idx} = data{nv_idx};
        end
        
    case 8
        lambdas = [0.2, 0.05, 0.2, 0.2];
        betas = [1, 1, 1, 1];

        filename = "Caltech101";
        load('Caltech101.mat');
        nv = size(fea, 2);
        gnd = gt';

        %We removed the background category.
        positions = find(gnd > 1);
        gnd = gnd(positions);
        K = length(unique(gnd));
        gnd = gnd - 1;

        data_views = cell(1, nv);
        for nv_idx = 1 : nv
             tmp = fea{nv_idx}';
             data_views{nv_idx} = tmp(:, positions);
        end
        n = length(gnd);
        new_dims = [200, 200, 200, 200];

   case 9
        lambdas = [0.4, 1.5, 0.2, 0.5];
        betas = [5, 5, 5, 5];

        filename = "scene";
        load('scene.mat');
        n = length(Y);
        nv = size(X, 2);
        K = length(unique(Y));
        gnd = Y;
        data_views = cell(1, nv);
        for nv_idx = 1 : nv
             data_views{nv_idx} = X{nv_idx}';
        end
end

%---------- The missing ratios --------------------------------------------
%- There are four experiments. --------------------------------------------
%--For example-------------------------------------------------------------
%- 0 represents that all features are available.
%- 0.1 represents that 10% of features are randomly missing in each view.
%-------------------------------------------------------------------------
missing_raitos = [0, 0.1, 0.3, 0.5];
ratio_len = length(missing_raitos);
repeated_times = 1; % for testing
% repeated_times = 10;
maxIter = 100;
final_result = strcat(filename, '_final_result.txt');
final_average_result = strcat(filename, '_final_average_result.txt');
final_convergence = strcat(filename, '_final_convergence.mat');

final_clustering_accs = zeros(ratio_len, repeated_times);
final_clustering_nmis = zeros(ratio_len, repeated_times);
final_clustering_purities = zeros(ratio_len, repeated_times);
final_clustering_fmeasures = zeros(ratio_len, repeated_times);
final_clustering_ris = zeros(ratio_len, repeated_times);
final_clustering_aris = zeros(ratio_len, repeated_times);
final_clustering_costs = zeros(ratio_len, repeated_times);
final_clustering_iters = zeros(ratio_len, repeated_times);
final_clustering_values = zeros(ratio_len,  maxIter);

Mn = cell(1, nv);
for raito_idx = 1 : length(missing_raitos)
    
    % a set of the incomplete data instances 
    stream = RandStream.getGlobalStream;
    reset(stream);
    missing_raito = missing_raitos(raito_idx);
    raito = 1 - missing_raito;    
    rand('state', 100);
    for nv_idx = 1 : nv        
        if raito < 1
            pos = randperm(n);
            num = floor(n * raito);
            sample_pos = zeros(1, n);
            % 1 represents that the corresponding features are available.
            sample_pos(pos(1 : num)) = 1; 
            Mn{nv_idx} = sample_pos;
        else
            Mn{nv_idx} = ones(1, n);
        end
    end

    dim = new_dims(raito_idx);
    lambda = lambdas(raito_idx);
    beta = betas(raito_idx);         
   
    tic;  
    [Hn, Ln, ~]  = cal_embedding_matrices(data_views, Mn, lambda, K, dim);                   
%     [F, H, iter, obj_values]  = orginal_lrtl(Hn, beta);
    [F, H, iter, obj_values]  = lrtl(Hn, beta); 
    final_clustering_values(raito_idx, 1 : maxIter) = obj_values(1 : maxIter);   
    time_cost = toc;
    
    for time_idx = 1 : repeated_times
        [acc, nmi, purity, fmeasure, ri, ari] = calculate_clustering_results_by_kmeans(F, gnd, K);               
        %[acc, nmi, purity, fmeasure, ri, ari] = calculate_clustering_results_by_kmeans(H, gnd, K);
        
        final_clustering_accs(raito_idx, time_idx) = acc;
        final_clustering_nmis(raito_idx, time_idx) = nmi;
        final_clustering_purities(raito_idx, time_idx) = purity;
        final_clustering_fmeasures(raito_idx, time_idx) = fmeasure;
        final_clustering_ris(raito_idx, time_idx) = ri;
        final_clustering_aris(raito_idx, time_idx) = ari;
        final_clustering_costs(raito_idx, time_idx) = time_cost;
        final_clustering_iters(raito_idx, time_idx) = iter;
        disp([missing_raito, time_idx, dim, lambda, beta, acc, nmi, purity, fmeasure, ri, ari, iter]);
        % Please umcomment the following statement to store data in a text file.
        %writematrix([missing_raito, time_idx, dim, lambda, beta, roundn(acc, -2), roundn(nmi, -4), roundn(purity, -4), roundn(fmeasure, -4), roundn(ri, -4), roundn(ari, -4), roundn(time_cost, -2), iter], final_result, "Delimiter", 'tab', 'WriteMode', 'append'); 
    end     

    averge_acc = mean(final_clustering_accs(raito_idx, :));
    std_acc = std(final_clustering_accs(raito_idx, :));
    averge_nmi = mean(final_clustering_nmis(raito_idx, :)); 
    std_nmi =std(final_clustering_nmis(raito_idx, :));
    averge_purity = mean(final_clustering_purities(raito_idx, :));
    std_purity =std(final_clustering_purities(raito_idx, :));
    averge_fmeasure = mean(final_clustering_fmeasures(raito_idx, :));
    std_fmeasure =std(final_clustering_fmeasures(raito_idx, :));
    averge_ri =  mean(final_clustering_ris(raito_idx, :));
    std_ri =std(final_clustering_ris(raito_idx, :));
    averge_ari = mean(final_clustering_aris(raito_idx, :));
    std_ari =std(final_clustering_aris(raito_idx, :));
    averge_cost = mean(final_clustering_costs(raito_idx, :)); 
    averge_iter = mean(final_clustering_iters(raito_idx, :));

    % Please umcomment the following statement to store data in a text file.
%      writematrix([missing_raito, dim, lambda, beta, roundn(averge_acc, -2), roundn(std_acc, -2), roundn(averge_nmi, -4), roundn(std_nmi, -4), ...
%          roundn(averge_purity, -4), roundn(std_purity, -4), roundn(averge_fmeasure, -4), roundn(std_fmeasure, -4), roundn(averge_ri, -4), roundn(std_ri, -4),...
%          roundn(averge_ari, -4), roundn(std_ari, -4), roundn(averge_cost, -2), roundn(averge_iter, -2)], final_average_result, "Delimiter", 'tab', 'WriteMode', 'append'); 
end
% save(final_convergence, 'final_clustering_values');
