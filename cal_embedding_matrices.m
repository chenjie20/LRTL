function [Hn, Ln, Wn]  = cal_embedding_matrices(data_views, Mn, lambda, k, dim)
 
    nv = length(data_views);
    n = size(data_views{1}, 2);
    cols = cell(1, nv);
    Wn = zeros(n, n, nv);
    Ln = zeros(n, n, nv);
    Hn = zeros(n, k, nv);
    
    for idx = 1 : nv
        cols{idx} = abs(Mn{idx} - 1) < 1e-6;
        X = data_views{idx}(:, cols{idx});
        if dim > 0 && dim < size(X, 1)
            [eigen_vector, ~] = f_pca(normc(X), dim);
            X = eigen_vector' *  X;        
        end
        X = normc(X);

        [~, s, V] = svd(X, 'econ');
        s = diag(s);
        lmd = 1 / sqrt(lambda);
        kk = length(find(s > lmd));
        Z = V(:, 1 : kk) * (eye(kk) -  diag(1 / (lambda * (s(1 : kk) .^ 2)))) * V(:, 1 : kk)';   

        [U, s, ~] = svd(Z, 'econ');
        s = diag(s);
        r = sum(s>1e-6);

        U = U(:, 1 : r);
        s = diag(s(1 : r));

        M = U * s.^(1/2);
        mm = normr(M);
        rs = mm * mm';
        Wn(cols{idx}, cols{idx}, idx) = rs.^2;
        
        W = Wn(:, :, idx) + eps;
        W = normc(W);
        W = project_simplex(W);
        W = (abs(W) + abs(W')) / 2;
        D = diag(1./sqrt(sum(W, 2)+ eps));
        Ln(:, :, idx) = D * W * D; 
        
        L = (Ln(:, :, idx) + Ln(:, :, idx)') / 2 + eps;
        [H, ~] = eigs(L, k, 'la');
        Hn(:, :, idx) = H;
        
    end
end

