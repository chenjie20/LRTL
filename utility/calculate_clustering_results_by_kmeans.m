function [acc, nmi, purity, fmeasure, ri, ari] = calculate_clustering_results_by_kmeans(U, groundLables, numClusters)

%         stream = RandStream.getGlobalStream;
%         reset(stream);
        U_normalized = U ./ repmat(sqrt(sum(U.^2, 2)), 1, numClusters);
        maxIter = 1000;
        currentLabels = litekmeans(U_normalized, numClusters, 'MaxIter',100, 'Replicates',maxIter);
                
        groundLables = groundLables(:);   
        currentLabels = currentLabels(:);
        
        acc = accuracy(groundLables, currentLabels);
        
        class_labels = zeros(1, numClusters);
        for idx =  1 : numClusters
            class_labels(idx) = length(find(groundLables == idx));
        end
        cluster_data = cell(1, numClusters);
        for idx =  1 : numClusters
            cluster_data(1, idx) = { groundLables(currentLabels == idx)' };
        end
        [nmi, purity, fmeasure, ri, ari] = calculate_results(class_labels, cluster_data);

end
