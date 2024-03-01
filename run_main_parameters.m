close all;
clear;
clc;

addpath('data');
addpath('utility');

%---------------------- parameters -------------------------
% lambdas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1, 1.2, 1.5, 2, 3, 5, 10, 15, 20, 30, 40, 50];
% betas = [0.1, 0.5, 1, 2, 5, 10, 15, 20, 25, 30, 40, 50];

lambdas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1, 1.2, 1.5, 2, 3, 5, 10, 20];
betas = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20];

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
        new_dims = [500, 300, 300, 100];

    case 3
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

     case 9
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

final_parameter = strcat(filename, '_final_parameters.txt');

%---------- The missing ratios --------------------------------------------
%- There are four experiments. --------------------------------------------
%--For example-------------------------------------------------------------
%- 0 represents that all features are available.
%- 0.1 represents that 10% of features are randomly missing in each view.
%-------------------------------------------------------------------------
missing_raitos = [0, 0.1, 0.3, 0.5];

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
    for lmd_idx = 1 : length(lambdas)      
        lambda = lambdas(lmd_idx);
        [Hn, Ln, ~]  = cal_embedding_matrices(data_views, Mn, lambda, K, dim);
        for beta_idx = 1 : length(betas)
            beta = betas(beta_idx);
            tic; 
%             [F, H, iter, ~]  = orginal_lrtl(Hn, beta);
            [F, H, iter, ~]  = lrtl(Hn, beta);             
            stream = RandStream.getGlobalStream;
            reset(stream);
            [acc, nmi, purity, fmeasure, ri, ari] = calculate_clustering_results_by_kmeans(F, gnd, K);
            time_cost = toc;
            disp([missing_raito, dim, lambda, beta, acc, nmi, purity, fmeasure, ri, ari, iter]);
            writematrix([missing_raito, dim, lambda, beta, roundn(acc, -2), roundn(nmi, -4), roundn(purity, -4), roundn(fmeasure, -4), roundn(ri, -4), roundn(ari, -4), roundn(time_cost, -2), iter], final_parameter, "Delimiter", 'tab', 'WriteMode', 'append');            
        end
    end
end
