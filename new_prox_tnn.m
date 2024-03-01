function X = new_prox_tnn(Y, rho)
% Y: d * 1 * n
% d: the dimensionality of each instance
% n: represents the number of instances

% References: 
% Canyi Lu, Jiashi Feng, Yudong Chen, Wei Liu, Zhouchen Lin and Shuicheng
% Yan, Tensor Robust Principal Component Analysis with A New Tensor Nuclear
% Norm

    [n1,n2,n3] = size(Y);
    X = zeros(n1,n2,n3);
    Y = fft(Y,[],3);

    % first frontal slice
    ynorm = norm(Y(:,1,1));
    if ynorm > rho  
        X(:,1,1) = svd_with_shrinkage_thresholding(Y(:,1,1), rho);     
        for nv_idx = 2 : n2
            X(:,nv_idx,1) = X(:,1,1);
        end
    end

    % i=2,...,halfn3
    halfn3 = round(n3/2);
    for i = 2 : halfn3
        ynorm = norm(Y(:,1,i));
        if ynorm > rho
            X(:,1,i) = svd_with_shrinkage_thresholding(Y(:,1,i), rho);    
            for nv_idx = 2 : n2
                X(:,nv_idx, i) = X(:,1, i);
            end
        end
        X(:,:,n3+2-i) = conj(X(:,:,i));
    end

    % if n3 is even
    if mod(n3,2) == 0
        i = halfn3 + 1;
        ynorm = norm(Y(:,1,i));
        if ynorm > rho      
            X(:,1,i) = svd_with_shrinkage_thresholding(Y(:,1,i), rho);        
            for nv_idx = 2 : n2
                X(:,nv_idx, i) = X(:,1, i);
            end
        end
    end
    X = ifft(X,[],3);
end

function Y = svd_with_shrinkage_thresholding(X, rho)
    xnorm = norm(X);
    U = X / xnorm;
    S = xnorm - rho;
    Y = U * S;
end