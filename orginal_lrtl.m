function [F, H_updated, iter, obj_values] = orginal_lrtl(Hn, beta)

mu = 1e-4;
mu_max = 1e6;
rho = 1.2;
iter = 0;
tol = 2e-6;
maxIter = 100;
obj_values = zeros(1, maxIter);

[num, k, nv] = size(Hn);

R = zeros(num, num, nv);
G = zeros(num, num, nv);
T = zeros(num, num, nv);

%Initialization
Hn_updated = zeros(num, k, nv);
for nv_idx = 1 : nv
   Hn_updated(:, :, nv_idx) = Hn(:, :, nv_idx);  
end

alphas = ones(nv, 1) * sqrt(1 / nv);
bHn = zeros(num, k); 
FHn_results = zeros(nv, 1);

while (iter < maxIter)

    iter = iter + 1;    
    %update F
    bHn(:, :) = 0;
    for nv_idx = 1 : nv
        bHn = bHn + alphas(nv_idx) * Hn_updated( : , : , nv_idx);
    end    
    [Uf, ~, Vf] = svd(bHn, 'econ');
    F = Uf * Vf';
    
    %optimize Hv
    A = G - 1 / mu * R;
    opts.info = 0;
    for nv_idx = 1 : nv
        Ht = Hn_updated(:, : , nv_idx);%
        At = A(:, :, nv_idx);
        Bt = - beta * alphas(nv_idx) * F;
        Ct = mu / 2 * ((Ht * Ht') - (At + At'));
        [Ht, ~] = FOForth(Ht, Bt, @fun, opts, Ct, Bt);
        Hn_updated(:, : , nv_idx) = Ht;
    end
    
    %optimize G
    for nv_idx = 1 : nv
        T( : , : , nv_idx) = Hn_updated( : , : , nv_idx) * Hn_updated( : , : , nv_idx)';
    end
    tempT1 = T + 1 / mu * R;
    tempT = shiftdim(tempT1, 1);
    [tempG, ~, ~] = prox_tnn(tempT, 1 / mu);
    G = shiftdim(tempG, 2);
    
    %update R and rho
    R = R + mu * (T - G);    
    mu = min(mu * rho, mu_max);
    
    %update beta
    for nv_idx = 1 : nv
        FHn_results(nv_idx) = trace(F' * Hn_updated( : , : , nv_idx));
    end
    alphas = FHn_results  / norm(FHn_results);
    
    %calculate the error
    err = max(max(max(abs(G - T))));
%     disp([iter, err]);
    obj_values(1, iter) = err;
    if err < tol
        break;
    end   
     
end
H_updated = reshape(Hn_updated( : , : , 1), num, k);
end

function [funX, F] = fun(X, A, Kp)
    F = 2 * A * X + Kp;
    funX = sum(sum(X .* (A * X + Kp)));
end

