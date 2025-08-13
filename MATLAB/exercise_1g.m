%% RBC Model – EGM (200 Points) with Optimization Tricks
% Part 1g : Greenwood, Hercowitz & Huffman (1988) with capacity utilisation
% Using EGM with 200 grid points implementing optimization tricks in VFI step
% (i) Policy function monotonicity and (ii) Concavity of maximand
% ------------------------------------------------------------------------

clearvars; close all; clc;

%% 1.  Parameters ---------------------------------------------------------
beta   = 0.96;      % Discount factor
alpha  = 0.33;      % Capital share
theta  = 1.0;       % Frisch elasticity parameter
gamma  = 2;         % Coefficient of relative risk aversion
B      = 0.075;     % Depreciation-usage scale
omega  = 2;         % Depreciation-usage curvature
A      = 0.592;     % Level of TFP

% Investment-specific shock ε_t ∈ {–Θ, +Θ},  two-state Markov
sigma_data = 0.051;           % std from data   => Θ
lambda     = 0.44;            % first-order serial corr. => π
Theta  = sigma_data;
pi_stay = (1+lambda)/2;       % transition prob.
P  = [ pi_stay  1-pi_stay ;      % 2×2 transition matrix
       1-pi_stay  pi_stay ];
eps_grid = [-Theta ; +Theta];

%% 2.  Grids (200 points as specified) -----------------------------------
Nk = 200;                      % # of k points (as required by exercise)
% crude steady-state guess (ignoring shocks & utilisation choice)
k_ss = ((1/beta - 1 + B)/(alpha*A))^(1/(alpha-1));
k_min = 0.25*k_ss;             % Same as exercise 1f for stability
k_max = 2.0*k_ss;              % Same as exercise 1f
k_grid = linspace(k_min,k_max,Nk)';

fprintf('=== Exercise 1g: EGM (200 Points) with Optimization Tricks ===\n');
fprintf('k_ss (crude guess) = %.3f\n', k_ss);
fprintf('Grid range: [%.3f, %.3f] (%.1f%% to %.1f%% of k_ss)\n', ...
        k_min, k_max, 100*k_min/k_ss, 100*k_max/k_ss);
fprintf('Implementing tricks in VFI step: (i) Monotonicity (ii) Concavity\n');

%% 3.  Pre-compute static decisions h*(k,ε) and ℓ*(k,ε) ------------------
fprintf('\n--- Pre-computing optimal h and ℓ for each (k,ε) state --------\n');
h_star = zeros(Nk,2);          % optimal capacity utilization
l_star = zeros(Nk,2);          % optimal labor

for ie = 1:2
    exp_eps = exp(-eps_grid(ie));
    for ik = 1:Nk
        k_now  = k_grid(ik);
        prod_k = A*(k_now)^alpha;           % constant term
        
        % FOC for h (after substituting optimal ℓ)
        hFOC = @(h) ...
            prod_k * alpha * h^(alpha-1) * ...
            ( ((1-alpha)*prod_k*h^alpha)^( (1-alpha)/(alpha+theta) ) ) ...
            - B * h^(omega-1) * k_now * exp_eps;
        
        % bracketing interval for fzero
        h_low = 0.05;  h_high = 3;
        if hFOC(h_low)*hFOC(h_high) < 0
            h_star(ik,ie) = fzero(hFOC,[h_low h_high]);
        else
            h_star(ik,ie) = 1;               % fallback (rare)
        end
        
        % corresponding ℓ*
        l_star(ik,ie) = ((1-alpha)*prod_k*h_star(ik,ie)^alpha)^(1/(alpha+theta));
    end
end
fprintf('Pre-computation complete.\n');

%% 4.  EGM Setup (Same as exercise 1f) -----------------------------------
fprintf('\n--- EGM Setup (Same Structure as Exercise 1f) ------------------\n');

% EGM endogenous grid for tomorrow's capital
kprime_min = k_min * 0.9;
kprime_max = k_max * 1.1; 
kprime_grid = linspace(kprime_min, kprime_max, Nk)';

% Policy functions on endogenous grid
c_endo = zeros(Nk, 2);        % consumption c(k',ε)
k_endo = zeros(Nk, 2);        % today's capital k(k',ε) that chooses k'

% Initialize value function
V = zeros(Nk, 2);
for ie = 1:2
    for ik = 1:Nk
        k = k_grid(ik);
        h = h_star(ik, ie);
        l = l_star(ik, ie);
        y = A * (k*h)^alpha * l^(1-alpha);
        c_guess = 0.6 * y;
        margin = c_guess - l^(1+theta)/(1+theta);
        if margin > 0
            V(ik, ie) = margin^(1-gamma)/(1-gamma) / (1-beta);
        else
            V(ik, ie) = -1000;
        end
    end
end

%% 5.  EGM with Different VFI Optimization Strategies -------------------
fprintf('\n--- Running EGM with Different VFI Optimization Strategies -----\n');

% EGM iteration parameters
tol = 1e-6;
max_iter = 100;  % Fewer iterations for timing test

% Storage for timing results
times = struct();

%% 5.1 BASELINE EGM (No Optimization Tricks)
fprintf('\nRunning Baseline EGM (no tricks in VFI step)...\n');
V_baseline = V;
tic;
for iter = 1:max_iter
    V_old = V_baseline;
    
    % EGM Steps 1-4 (same for all methods)
    for ie = 1:2
        exp_eps = exp(-eps_grid(ie));
        
        for ikp = 1:Nk
            kprime = kprime_grid(ikp);
            
            % Compute expected marginal utility tomorrow
            Emu_c_R = 0;
            for ie_next = 1:2
                prob = P(ie, ie_next);
                if kprime >= k_min && kprime <= k_max
                    h_tom = interp1(k_grid, h_star(:, ie_next), kprime, 'linear', 'extrap');
                    l_tom = interp1(k_grid, l_star(:, ie_next), kprime, 'linear', 'extrap');
                    delta_tom = B * h_tom^omega / omega;
                    y_tom = A * (kprime * h_tom)^alpha * l_tom^(1-alpha);
                    
                    if iter == 1
                        c_tom = 0.6 * y_tom;
                    else
                        c_tom = interp1(kprime_grid, c_endo(:, ie_next), kprime, 'linear', 'extrap');
                        c_tom = max(c_tom, 0.05 * y_tom);
                        c_tom = min(c_tom, 0.9 * y_tom);
                    end
                    
                    margin_tom = c_tom - l_tom^(1+theta)/(1+theta);
                    if margin_tom > 1e-12
                        mu_c_tom = margin_tom^(-gamma);
                        MPK_tom = alpha * y_tom / kprime;
                        R_tom = MPK_tom + (1 - delta_tom);
                        Emu_c_R = Emu_c_R + prob * mu_c_tom * R_tom;
                    end
                end
            end
            
            % Find today's consumption from Euler equation
            if Emu_c_R > 1e-12
                target_mu_c = beta * Emu_c_R;
                target_margin = target_mu_c^(-1/gamma);
                
                best_k = k_ss;
                best_c = 0.1;
                min_error = Inf;
                
                for ik = 1:Nk
                    k_try = k_grid(ik);
                    h_try = h_star(ik, ie);
                    l_try = l_star(ik, ie);
                    delta_try = B * h_try^omega / omega;
                    y_try = A * (k_try * h_try)^alpha * l_try^(1-alpha);
                    
                    min_kprime = k_try * (1 - delta_try) * exp_eps;
                    if kprime < min_kprime
                        continue;
                    end
                    
                    c_implied = y_try + k_try*(1-delta_try)*exp_eps - kprime*exp_eps;
                    
                    if c_implied > 1e-10
                        margin_implied = c_implied - l_try^(1+theta)/(1+theta);
                        if margin_implied > 1e-10
                            error = abs(margin_implied - target_margin);
                            if error < min_error
                                min_error = error;
                                best_k = k_try;
                                best_c = c_implied;
                            end
                        end
                    end
                end
                
                c_endo(ikp, ie) = best_c;
                k_endo(ikp, ie) = best_k;
            else
                c_endo(ikp, ie) = 0.1;
                k_endo(ikp, ie) = k_ss;
            end
        end
    end
    
    % STEP 5: VFI Update (BASELINE - No tricks)
    V_new = zeros(Nk, 2);
    for ie = 1:2
        exp_eps = exp(-eps_grid(ie));
        for ik = 1:Nk
            k_now = k_grid(ik);
            h_opt = h_star(ik, ie);
            l_opt = l_star(ik, ie);
            y = A*(k_now*h_opt)^alpha * l_opt^(1-alpha);
            delta = B*h_opt^omega/omega;
            
            % BASELINE: Full search, no tricks
            candV = -Inf(Nk,1);
            for ikp = 1:Nk
                k_next = k_grid(ikp);
                
                if k_next < k_now*(1-delta)*exp_eps
                    continue;
                end
                
                c = y - k_next*exp_eps + k_now*(1-delta)*exp_eps;
                margin = c - l_opt^(1+theta)/(1+theta);
                
                if margin > 1e-10 && l_opt>0 && l_opt<1
                    util = margin^(1-gamma)/(1-gamma);
                    EV = P(ie,:)*V_baseline(ikp,:)';
                    candV(ikp) = util + beta*EV;
                elseif margin <= 1e-10 && gamma>1
                    candV(ikp) = -Inf;
                end
            end
            
            [V_new(ik,ie),~] = max(candV);
        end
    end
    
    V_baseline = V_new;
    if iter == max_iter
        break;
    end
end
times.baseline = toc;
fprintf('Baseline EGM: %.4f seconds (%d iterations)\n', times.baseline, max_iter);

%% 5.2 EGM with Trick 1: Monotonicity Only
fprintf('\nRunning EGM with Trick 1 (Monotonicity only)...\n');
V_mono = V;
tic;
for iter = 1:max_iter
    V_old = V_mono;
    
    % EGM Steps 1-4 (same as baseline)
    for ie = 1:2
        exp_eps = exp(-eps_grid(ie));
        
        for ikp = 1:Nk
            kprime = kprime_grid(ikp);
            
            Emu_c_R = 0;
            for ie_next = 1:2
                prob = P(ie, ie_next);
                if kprime >= k_min && kprime <= k_max
                    h_tom = interp1(k_grid, h_star(:, ie_next), kprime, 'linear', 'extrap');
                    l_tom = interp1(k_grid, l_star(:, ie_next), kprime, 'linear', 'extrap');
                    delta_tom = B * h_tom^omega / omega;
                    y_tom = A * (kprime * h_tom)^alpha * l_tom^(1-alpha);
                    
                    if iter == 1
                        c_tom = 0.6 * y_tom;
                    else
                        c_tom = interp1(kprime_grid, c_endo(:, ie_next), kprime, 'linear', 'extrap');
                        c_tom = max(c_tom, 0.05 * y_tom);
                        c_tom = min(c_tom, 0.9 * y_tom);
                    end
                    
                    margin_tom = c_tom - l_tom^(1+theta)/(1+theta);
                    if margin_tom > 1e-12
                        mu_c_tom = margin_tom^(-gamma);
                        MPK_tom = alpha * y_tom / kprime;
                        R_tom = MPK_tom + (1 - delta_tom);
                        Emu_c_R = Emu_c_R + prob * mu_c_tom * R_tom;
                    end
                end
            end
            
            if Emu_c_R > 1e-12
                target_mu_c = beta * Emu_c_R;
                target_margin = target_mu_c^(-1/gamma);
                
                best_k = k_ss;
                best_c = 0.1;
                min_error = Inf;
                
                for ik = 1:Nk
                    k_try = k_grid(ik);
                    h_try = h_star(ik, ie);
                    l_try = l_star(ik, ie);
                    delta_try = B * h_try^omega / omega;
                    y_try = A * (k_try * h_try)^alpha * l_try^(1-alpha);
                    
                    min_kprime = k_try * (1 - delta_try) * exp_eps;
                    if kprime < min_kprime
                        continue;
                    end
                    
                    c_implied = y_try + k_try*(1-delta_try)*exp_eps - kprime*exp_eps;
                    
                    if c_implied > 1e-10
                        margin_implied = c_implied - l_try^(1+theta)/(1+theta);
                        if margin_implied > 1e-10
                            error = abs(margin_implied - target_margin);
                            if error < min_error
                                min_error = error;
                                best_k = k_try;
                                best_c = c_implied;
                            end
                        end
                    end
                end
                
                c_endo(ikp, ie) = best_c;
                k_endo(ikp, ie) = best_k;
            else
                c_endo(ikp, ie) = 0.1;
                k_endo(ikp, ie) = k_ss;
            end
        end
    end
    
    % STEP 5: VFI Update with TRICK 1 (Monotonicity)
    V_new = zeros(Nk, 2);
    for ie = 1:2
        exp_eps = exp(-eps_grid(ie));
        
        % TRICK 1: Track optimal indices
        best_idx_prev = 1;
        
        for ik = 1:Nk
            k_now = k_grid(ik);
            h_opt = h_star(ik, ie);
            l_opt = l_star(ik, ie);
            y = A*(k_now*h_opt)^alpha * l_opt^(1-alpha);
            delta = B*h_opt^omega/omega;
            
            % TRICK 1: Start search from previous optimal index
            search_start = max(1, best_idx_prev);
            
            % Find feasible starting point
            while search_start <= Nk && k_grid(search_start) < k_now*(1-delta)*exp_eps
                search_start = search_start + 1;
            end
            
            if search_start > Nk
                search_start = Nk;
            end
            
            best_value = -Inf;
            best_idx = search_start;
            
            % Search from starting point onwards
            for ikp = search_start:Nk
                k_next = k_grid(ikp);
                
                if k_next < k_now*(1-delta)*exp_eps
                    continue;
                end
                
                c = y - k_next*exp_eps + k_now*(1-delta)*exp_eps;
                margin = c - l_opt^(1+theta)/(1+theta);
                
                if margin > 1e-10 && l_opt>0 && l_opt<1
                    util = margin^(1-gamma)/(1-gamma);
                    EV = P(ie,:)*V_mono(ikp,:)';
                    value = util + beta*EV;
                    
                    if value > best_value
                        best_value = value;
                        best_idx = ikp;
                    end
                end
            end
            
            V_new(ik, ie) = best_value;
            best_idx_prev = best_idx;  % Update for monotonicity
        end
    end
    
    V_mono = V_new;
    if iter == max_iter
        break;
    end
end
times.monotonicity = toc;
fprintf('Monotonicity EGM: %.4f seconds (%d iterations)\n', times.monotonicity, max_iter);

%% 5.3 EGM with Trick 2: Concavity Only
fprintf('\nRunning EGM with Trick 2 (Concavity only)...\n');
V_concave = V;
tic;
for iter = 1:max_iter
    V_old = V_concave;
    
    % EGM Steps 1-4 (same as baseline)
    for ie = 1:2
        exp_eps = exp(-eps_grid(ie));
        
        for ikp = 1:Nk
            kprime = kprime_grid(ikp);
            
            Emu_c_R = 0;
            for ie_next = 1:2
                prob = P(ie, ie_next);
                if kprime >= k_min && kprime <= k_max
                    h_tom = interp1(k_grid, h_star(:, ie_next), kprime, 'linear', 'extrap');
                    l_tom = interp1(k_grid, l_star(:, ie_next), kprime, 'linear', 'extrap');
                    delta_tom = B * h_tom^omega / omega;
                    y_tom = A * (kprime * h_tom)^alpha * l_tom^(1-alpha);
                    
                    if iter == 1
                        c_tom = 0.6 * y_tom;
                    else
                        c_tom = interp1(kprime_grid, c_endo(:, ie_next), kprime, 'linear', 'extrap');
                        c_tom = max(c_tom, 0.05 * y_tom);
                        c_tom = min(c_tom, 0.9 * y_tom);
                    end
                    
                    margin_tom = c_tom - l_tom^(1+theta)/(1+theta);
                    if margin_tom > 1e-12
                        mu_c_tom = margin_tom^(-gamma);
                        MPK_tom = alpha * y_tom / kprime;
                        R_tom = MPK_tom + (1 - delta_tom);
                        Emu_c_R = Emu_c_R + prob * mu_c_tom * R_tom;
                    end
                end
            end
            
            if Emu_c_R > 1e-12
                target_mu_c = beta * Emu_c_R;
                target_margin = target_mu_c^(-1/gamma);
                
                best_k = k_ss;
                best_c = 0.1;
                min_error = Inf;
                
                for ik = 1:Nk
                    k_try = k_grid(ik);
                    h_try = h_star(ik, ie);
                    l_try = l_star(ik, ie);
                    delta_try = B * h_try^omega / omega;
                    y_try = A * (k_try * h_try)^alpha * l_try^(1-alpha);
                    
                    min_kprime = k_try * (1 - delta_try) * exp_eps;
                    if kprime < min_kprime
                        continue;
                    end
                    
                    c_implied = y_try + k_try*(1-delta_try)*exp_eps - kprime*exp_eps;
                    
                    if c_implied > 1e-10
                        margin_implied = c_implied - l_try^(1+theta)/(1+theta);
                        if margin_implied > 1e-10
                            error = abs(margin_implied - target_margin);
                            if error < min_error
                                min_error = error;
                                best_k = k_try;
                                best_c = c_implied;
                            end
                        end
                    end
                end
                
                c_endo(ikp, ie) = best_c;
                k_endo(ikp, ie) = best_k;
            else
                c_endo(ikp, ie) = 0.1;
                k_endo(ikp, ie) = k_ss;
            end
        end
    end
    
    % STEP 5: VFI Update with TRICK 2 (Concavity)
    V_new = zeros(Nk, 2);
    for ie = 1:2
        exp_eps = exp(-eps_grid(ie));
        for ik = 1:Nk
            k_now = k_grid(ik);
            h_opt = h_star(ik, ie);
            l_opt = l_star(ik, ie);
            y = A*(k_now*h_opt)^alpha * l_opt^(1-alpha);
            delta = B*h_opt^omega/omega;
            
            best_value = -Inf;
            
            % TRICK 2: Early stopping based on concavity
            for ikp = 1:Nk
                k_next = k_grid(ikp);
                
                if k_next < k_now*(1-delta)*exp_eps
                    continue;
                end
                
                c = y - k_next*exp_eps + k_now*(1-delta)*exp_eps;
                margin = c - l_opt^(1+theta)/(1+theta);
                
                if margin > 1e-10 && l_opt>0 && l_opt<1
                    util = margin^(1-gamma)/(1-gamma);
                    EV = P(ie,:)*V_concave(ikp,:)';
                    value = util + beta*EV;
                    
                    if value > best_value
                        best_value = value;
                    else
                        % TRICK 2: Stop when value decreases (concavity)
                        if ikp > 10  % Allow some tolerance
                            break;
                        end
                    end
                end
            end
            
            V_new(ik, ie) = best_value;
        end
    end
    
    V_concave = V_new;
    if iter == max_iter
        break;
    end
end
times.concavity = toc;
fprintf('Concavity EGM: %.4f seconds (%d iterations)\n', times.concavity, max_iter);

%% 5.4 EGM with Both Tricks
fprintf('\nRunning EGM with Both Tricks...\n');
V_both = V;
tic;
for iter = 1:max_iter
    V_old = V_both;
    
    % EGM Steps 1-4 (same as baseline)
    for ie = 1:2
        exp_eps = exp(-eps_grid(ie));
        
        for ikp = 1:Nk
            kprime = kprime_grid(ikp);
            
            Emu_c_R = 0;
            for ie_next = 1:2
                prob = P(ie, ie_next);
                if kprime >= k_min && kprime <= k_max
                    h_tom = interp1(k_grid, h_star(:, ie_next), kprime, 'linear', 'extrap');
                    l_tom = interp1(k_grid, l_star(:, ie_next), kprime, 'linear', 'extrap');
                    delta_tom = B * h_tom^omega / omega;
                    y_tom = A * (kprime * h_tom)^alpha * l_tom^(1-alpha);
                    
                    if iter == 1
                        c_tom = 0.6 * y_tom;
                    else
                        c_tom = interp1(kprime_grid, c_endo(:, ie_next), kprime, 'linear', 'extrap');
                        c_tom = max(c_tom, 0.05 * y_tom);
                        c_tom = min(c_tom, 0.9 * y_tom);
                    end
                    
                    margin_tom = c_tom - l_tom^(1+theta)/(1+theta);
                    if margin_tom > 1e-12
                        mu_c_tom = margin_tom^(-gamma);
                        MPK_tom = alpha * y_tom / kprime;
                        R_tom = MPK_tom + (1 - delta_tom);
                        Emu_c_R = Emu_c_R + prob * mu_c_tom * R_tom;
                    end
                end
            end
            
            if Emu_c_R > 1e-12
                target_mu_c = beta * Emu_c_R;
                target_margin = target_mu_c^(-1/gamma);
                
                best_k = k_ss;
                best_c = 0.1;
                min_error = Inf;
                
                for ik = 1:Nk
                    k_try = k_grid(ik);
                    h_try = h_star(ik, ie);
                    l_try = l_star(ik, ie);
                    delta_try = B * h_try^omega / omega;
                    y_try = A * (k_try * h_try)^alpha * l_try^(1-alpha);
                    
                    min_kprime = k_try * (1 - delta_try) * exp_eps;
                    if kprime < min_kprime
                        continue;
                    end
                    
                    c_implied = y_try + k_try*(1-delta_try)*exp_eps - kprime*exp_eps;
                    
                    if c_implied > 1e-10
                        margin_implied = c_implied - l_try^(1+theta)/(1+theta);
                        if margin_implied > 1e-10
                            error = abs(margin_implied - target_margin);
                            if error < min_error
                                min_error = error;
                                best_k = k_try;
                                best_c = c_implied;
                            end
                        end
                    end
                end
                
                c_endo(ikp, ie) = best_c;
                k_endo(ikp, ie) = best_k;
            else
                c_endo(ikp, ie) = 0.1;
                k_endo(ikp, ie) = k_ss;
            end
        end
    end
    
    % STEP 5: VFI Update with BOTH TRICKS
    V_new = zeros(Nk, 2);
    for ie = 1:2
        exp_eps = exp(-eps_grid(ie));
        
        % TRICK 1: Track optimal indices
        best_idx_prev = 1;
        
        for ik = 1:Nk
            k_now = k_grid(ik);
            h_opt = h_star(ik, ie);
            l_opt = l_star(ik, ie);
            y = A*(k_now*h_opt)^alpha * l_opt^(1-alpha);
            delta = B*h_opt^omega/omega;
            
            % TRICK 1: Start from previous optimal index
            search_start = max(1, best_idx_prev);
            while search_start <= Nk && k_grid(search_start) < k_now*(1-delta)*exp_eps
                search_start = search_start + 1;
            end
            if search_start > Nk
                search_start = Nk;
            end
            
            best_value = -Inf;
            best_idx = search_start;
            
            % BOTH TRICKS: Monotonic start + early stopping
            for ikp = search_start:Nk
                k_next = k_grid(ikp);
                
                if k_next < k_now*(1-delta)*exp_eps
                    continue;
                end
                
                c = y - k_next*exp_eps + k_now*(1-delta)*exp_eps;
                margin = c - l_opt^(1+theta)/(1+theta);
                
                if margin > 1e-10 && l_opt>0 && l_opt<1
                    util = margin^(1-gamma)/(1-gamma);
                    EV = P(ie,:)*V_both(ikp,:)';
                    value = util + beta*EV;
                    
                    if value > best_value
                        best_value = value;
                        best_idx = ikp;
                    else
                        % TRICK 2: Early stopping
                        if ikp > search_start + 5
                            break;
                        end
                    end
                end
            end
            
            V_new(ik, ie) = best_value;
            best_idx_prev = best_idx;
        end
    end
    
    V_both = V_new;
    if iter == max_iter
        break;
    end
end
times.both = toc;
fprintf('Both tricks EGM: %.4f seconds (%d iterations)\n', times.both, max_iter);

%% 6.  Extract Final Policy Functions (using both tricks method) ---------
fprintf('\n--- Extracting Final Policy Functions (Both Tricks Method) -----\n');

Kpol = zeros(Nk, 2);
for ie = 1:2
    exp_eps = exp(-eps_grid(ie));
    best_idx_prev = 1;
    
    for ik = 1:Nk
        k_now = k_grid(ik);
        h_opt = h_star(ik, ie);
        l_opt = l_star(ik, ie);
        y = A*(k_now*h_opt)^alpha * l_opt^(1-alpha);
        delta = B*h_opt^omega/omega;
        
        search_start = max(1, best_idx_prev);
        while search_start <= Nk && k_grid(search_start) < k_now*(1-delta)*exp_eps
            search_start = search_start + 1;
        end
        if search_start > Nk
            search_start = Nk;
        end
        
        best_value = -Inf;
        best_kprime = k_now;
        best_idx = search_start;
        
        for ikp = search_start:Nk
            k_next = k_grid(ikp);
            
            if k_next < k_now*(1-delta)*exp_eps
                continue;
            end
            
            c = y - k_next*exp_eps + k_now*(1-delta)*exp_eps;
            margin = c - l_opt^(1+theta)/(1+theta);
            
            if margin > 1e-10 && l_opt>0 && l_opt<1
                util = margin^(1-gamma)/(1-gamma);
                EV = P(ie,:)*V_both(ikp,:)';
                value = util + beta*EV;
                
                if value > best_value
                    best_value = value;
                    best_kprime = k_next;
                    best_idx = ikp;
                else
                    if ikp > search_start + 5
                        break;
                    end
                end
            end
        end
        
        Kpol(ik, ie) = best_kprime;
        best_idx_prev = best_idx;
    end
end

%% 7.  Performance Analysis ----------------------------------------------
fprintf('\n=== Performance Analysis ===\n');

speedup_mono = times.baseline / times.monotonicity;
speedup_concave = times.baseline / times.concavity;
speedup_both = times.baseline / times.both;

savings_mono = times.baseline - times.monotonicity;
savings_concave = times.baseline - times.concavity;
savings_both = times.baseline - times.both;

fprintf('\nTime for %d iterations:\n', max_iter);
fprintf('  Baseline EGM (no tricks):    %.4f seconds\n', times.baseline);
fprintf('  Monotonicity trick only:     %.4f seconds\n', times.monotonicity);
fprintf('  Concavity trick only:        %.4f seconds\n', times.concavity);
fprintf('  Both tricks combined:        %.4f seconds\n', times.both);

fprintf('\nSpeedup factors:\n');
fprintf('  Monotonicity trick:          %.1fx faster\n', speedup_mono);
fprintf('  Concavity trick:             %.1fx faster\n', speedup_concave);
fprintf('  Both tricks combined:        %.1fx faster\n', speedup_both);

fprintf('\nTime saved:\n');
fprintf('  Monotonicity trick:          %.4f seconds (%.1f%% reduction)\n', ...
        savings_mono, 100*savings_mono/times.baseline);
fprintf('  Concavity trick:             %.4f seconds (%.1f%% reduction)\n', ...
        savings_concave, 100*savings_concave/times.baseline);
fprintf('  Both tricks combined:        %.4f seconds (%.1f%% reduction)\n', ...
        savings_both, 100*savings_both/times.baseline);

%% 8.  Economic Analysis -------------------------------------------------
fprintf('\n=== Economic Diagnostics ===\n');

% Find steady states
[~, ss_idx_bad] = min(abs(Kpol(:,1) - k_grid));
[~, ss_idx_good] = min(abs(Kpol(:,2) - k_grid));

fprintf('\n--- Approximate Steady States ---\n');
fprintf('Bad shock (ε = %.3f):\n', eps_grid(1));
fprintf('  k = %.3f, k'' = %.3f, ratio k''/k = %.3f\n', ...
        k_grid(ss_idx_bad), Kpol(ss_idx_bad,1), Kpol(ss_idx_bad,1)/k_grid(ss_idx_bad));
fprintf('Good shock (ε = %.3f):\n', eps_grid(2));
fprintf('  k = %.3f, k'' = %.3f, ratio k''/k = %.3f\n', ...
        k_grid(ss_idx_good), Kpol(ss_idx_good,2), Kpol(ss_idx_good,2)/k_grid(ss_idx_good));

% Analyze depreciation
fprintf('\n--- Depreciation Analysis ---\n');
for ie = 1:2
    h = h_star(ss_idx_bad, ie);
    delta = B*h^omega/omega;
    fprintf('State ε = %.3f:\n', eps_grid(ie));
    fprintf('  Capacity utilization h = %.3f\n', h);
    fprintf('  Depreciation rate δ(h) = %.3f (%.1f%% per period)\n', delta, 100*delta);
    fprintf('  Survival rate 1-δ(h) = %.3f\n', 1-delta);
end

%% 9.  Plots -------------------------------------------------------------
figure(1)
plot(k_grid, V_both(:,1), 'b-', 'LineWidth', 1.6); hold on
plot(k_grid, V_both(:,2), 'r--', 'LineWidth', 1.6);
xlabel('Capital k','FontSize',11); 
ylabel('Value function V','FontSize',11);
legend('\epsilon = -\Theta','\epsilon = +\Theta','Location','SouthEast');
title('Value function - EGM (200 points) with optimization tricks','FontSize',12); 
grid on;
set(gca,'FontSize',10);

figure(2)
plot(k_grid, Kpol(:,1), 'b-', 'LineWidth', 1.6); hold on
plot(k_grid, Kpol(:,2), 'r--', 'LineWidth', 1.6);
plot(k_grid, k_grid, 'k:', 'LineWidth', 1);
xlabel('Current Capital k','FontSize',11); 
ylabel('Next-period capital k''','FontSize',11);
legend('\epsilon = -\Theta','\epsilon = +\Theta','45° line','Location','NorthWest');
title('Policy function - EGM (200 points) with optimization tricks','FontSize',12); 
grid on;
set(gca,'FontSize',10);

% Mark steady states
plot(k_grid(ss_idx_bad), Kpol(ss_idx_bad,1), 'bo', 'MarkerSize', 8);
plot(k_grid(ss_idx_good), Kpol(ss_idx_good,2), 'ro', 'MarkerSize', 8);

% Add performance info
xlim_curr = xlim; ylim_curr = ylim;
text(xlim_curr(1) + 0.02*diff(xlim_curr), ylim_curr(2) - 0.05*diff(ylim_curr), ...
     sprintf('Combined speedup: %.1fx', speedup_both), ...
     'FontSize',9,'Color',[0.5 0.5 0.5]);

%% 10. Save workspace ----------------------------------------------------
save rbc_egm_200_optimization_tricks.mat V_both Kpol k_grid eps_grid times ...
     speedup_mono speedup_concave speedup_both;
fprintf('\nResults saved to rbc_egm_200_optimization_tricks.mat\n');