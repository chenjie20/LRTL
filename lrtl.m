function [F, H_updated, iter, obj_values] = lrtl(Hn, beta)

mu = 1e-4;
mu_max = 1e6;
rho = 1.2;
iter = 0;
tol = 1e-6;
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

alphas = ones(nv, 1);
bHn = zeros(num, k); 

while (iter < maxIter)
    iter = iter + 1;        
    %update F
    if iter > 1
         F = (Hn_updated( : , : , nv_idx));
    else        
        bHn(:, :) = 0;
        for nv_idx = 1 : nv
            bHn = bHn + alphas(nv_idx) * Hn_updated( : , : , nv_idx);
        end    
        [Uf, ~, Vf] = svd(bHn, 'econ');
        F = Uf * Vf';
    end
    
%     err1 = max(max(abs(F) - abs((reshape(Hn_updated( : , : , 1), 1600, 100)))));
%     err2 = max(max(abs(F) - abs((reshape(Hn_updated( : , : , 2), 1600, 100)))));
%     disp([iter, err1, err2]);
    
    %optimize Hv
    A = G - 1 / mu * R;
    opts.info = 0;
    Ht = Hn_updated(:, : , 1);%
    At = A(:, :, 1);
    Bt = - beta * alphas(1) * F;
    Ct = mu / 2 * ((Ht * Ht') - (At + At'));
    [Ht, ~] = FOForth(Ht, Bt, @fun, opts, Ct, Bt);
    Hn_updated(:, : , 1) = Ht;    
    for nv_idx = 2 : nv
        Hn_updated(:, : , nv_idx) = Hn_updated(:, : , 1);
    end
     
    %optimize G
    T( : , : , 1) = Hn_updated( : , : , 1) * Hn_updated( : , : , 1)';
    for nv_idx = 2 : nv
        T( : , : , nv_idx) = T( : , : , 1);
    end
    tempT1 = T + 1 / mu * R;
    tempT = shiftdim(tempT1, 1);
    tempG = new_prox_tnn(tempT, 1 / mu);
    G = shiftdim(tempG, 2);

    %update R and rho
    R = R + mu * (T - G);    
    mu = min(mu * rho, mu_max);
        
    %calculate the error
    err = max(max(max(abs(G - T))));
    obj_values(1, iter) = err;
%     disp([iter, err]);
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

