%% RBC Model – Value-Function Iteration (Corrected Version)
% Part 1b : Greenwood, Hercowitz & Huffman (1988) with capacity utilisation
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

%% 2.  Grids --------------------------------------------------------------
Nk = 100;                      % # of k points
% crude steady-state guess (ignoring shocks & utilisation choice)
k_ss = ((1/beta - 1 + B)/(alpha*A))^(1/(alpha-1));
k_min = 0.20*k_ss;             % Start at 20% of steady state - completely avoids artifacts
k_max = 2.0*k_ss;              % Expanded upper bound to ensure no binding
k_grid = linspace(k_min,k_max,Nk)';

fprintf('=== Grid Setup ===\n');
fprintf('k_ss (crude guess) = %.3f\n', k_ss);
fprintf('Grid range: [%.3f, %.3f] (%.1f%% to %.1f%% of k_ss)\n', ...
        k_min, k_max, 100*k_min/k_ss, 100*k_max/k_ss);
fprintf('Note: k_min = 20%% of k_ss completely eliminates grid artifacts\n');

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

%% 4.  Containers & convergence params -----------------------------------
V       = zeros(Nk,2);         % value function
Vnew    = V;
Kpol    = zeros(Nk,2);         % optimal k'
tol     = 1e-6;  err = 1;  iter = 0;  max_iter = 1000;

%% 5.  Value-Function Iteration -------------------------------------------
fprintf('\n--- Value-Function Iteration --------------------------------\n');
tic;

while err>tol && iter<max_iter
    iter = iter + 1;
    for ie = 1:2                              % current ε index
        exp_eps = exp(-eps_grid(ie));
        for ik = 1:Nk                         % current k
            k_now  = k_grid(ik);
            
            % Pre-computed values for this state
            h_opt  = h_star(ik,ie);
            l_opt  = l_star(ik,ie);
            
            % Production and CORRECT depreciation
            y      = A*(k_now*h_opt)^alpha * l_opt^(1-alpha);
            delta  = B*h_opt^omega/omega;     % CORRECTED: δ(h) = B*h^ω/ω
            
            % build return + continuation value for every k'
            candV = -Inf(Nk,1);
            for ikp = 1:Nk
                k_next = k_grid(ikp);
                
                % INVESTMENT NON-NEGATIVITY CHECK
                % Ensure k' >= k(1-δ)e^{-ε} to prevent negative investment
                % Note: This constraint can create grid artifacts at very low k
                % because it forces k' to be at least ~95% of current k,
                % making it impossible to choose low k' when k is small.
                if k_next < k_now*(1-delta)*exp_eps
                    continue;  % skip this k' as it violates non-negativity
                end
                
                % Resource constraint with CORRECT depreciation
                c = y - k_next*exp_eps + k_now*(1-delta)*exp_eps;
                
                % Feasibility & utility
                margin = c - l_opt^(1+theta)/(1+theta);
                
                if margin > 1e-12 && l_opt>0 && l_opt<1   % guard against margin≈0
                    util = margin^(1-gamma)/(1-gamma);
                    
                    % expected continuation
                    EV = P(ie,:)*V(ikp,:)';
                    candV(ikp) = util + beta*EV;
                elseif margin <= 1e-12 && gamma>1
                    candV(ikp) = -Inf;                     % utility → -∞
                end
            end
            
            % maximise over k'
            [Vnew(ik,ie),best] = max(candV);
            Kpol(ik,ie)       = k_grid(best);
        end
    end
    
    err = max(abs(Vnew(:)-V(:)));
    V   = Vnew;
    if mod(iter,10)==0 || err<tol
        fprintf('Iter %3d   err = %.3e\n',iter,err);
    end
end

elapsed = toc;
fprintf('Converged in %d iterations (tol=%g) in %.2f seconds\n',iter,tol,elapsed);

% Check if hitting upper grid bound and auto-expand if needed
upper_bound_hit = any(Kpol(:,1)==k_max) || any(Kpol(:,2)==k_max);
if upper_bound_hit
    fprintf('\n*** WARNING: Policy hits upper bound k_max = %.3f ***\n', k_max);
    fprintf('*** Auto-expanding grid and re-running... ***\n\n');
    
    % Expand grid and re-run
    k_max = 2.5*k_ss;
    k_grid = linspace(k_min,k_max,Nk)';
    
    % Reset containers
    V = zeros(Nk,2);
    Vnew = V;
    err = 1; iter = 0;
    
    % Re-compute h* and l* for new grid
    fprintf('--- Re-computing h and l for expanded grid --------\n');
    for ie = 1:2
        exp_eps = exp(-eps_grid(ie));
        for ik = 1:Nk
            k_now  = k_grid(ik);
            prod_k = A*(k_now)^alpha;
            
            hFOC = @(h) ...
                prod_k * alpha * h^(alpha-1) * ...
                ( ((1-alpha)*prod_k*h^alpha)^( (1-alpha)/(alpha+theta) ) ) ...
                - B * h^(omega-1) * k_now * exp_eps;
            
            h_low = 0.05;  h_high = 3;
            if hFOC(h_low)*hFOC(h_high) < 0
                h_star(ik,ie) = fzero(hFOC,[h_low h_high]);
            else
                h_star(ik,ie) = 1;
            end
            
            l_star(ik,ie) = ((1-alpha)*prod_k*h_star(ik,ie)^alpha)^(1/(alpha+theta));
        end
    end
    
    % Re-run value function iteration
    fprintf('\n--- Re-running with expanded grid (k_max = %.3f) ---\n', k_max);
    while err>tol && iter<max_iter
        iter = iter + 1;
        for ie = 1:2
            exp_eps = exp(-eps_grid(ie));
            for ik = 1:Nk
                k_now  = k_grid(ik);
                h_opt  = h_star(ik,ie);
                l_opt  = l_star(ik,ie);
                y      = A*(k_now*h_opt)^alpha * l_opt^(1-alpha);
                delta  = B*h_opt^omega/omega;
                
                candV = -Inf(Nk,1);
                for ikp = 1:Nk
                    k_next = k_grid(ikp);
                    
                    if k_next < k_now*(1-delta)*exp_eps
                        continue;
                    end
                    
                    c = y - k_next*exp_eps + k_now*(1-delta)*exp_eps;
                    margin = c - l_opt^(1+theta)/(1+theta);
                    
                    if margin > 1e-12 && l_opt>0 && l_opt<1
                        util = margin^(1-gamma)/(1-gamma);
                        EV = P(ie,:)*V(ikp,:)';
                        candV(ikp) = util + beta*EV;
                    elseif margin <= 1e-12 && gamma>1
                        candV(ikp) = -Inf;
                    end
                end
                
                [Vnew(ik,ie),best] = max(candV);
                Kpol(ik,ie) = k_grid(best);
            end
        end
        
        err = max(abs(Vnew(:)-V(:)));
        V = Vnew;
        if mod(iter,10)==0 || err<tol
            fprintf('Iter %3d   err = %.3e\n',iter,err);
        end
    end
    fprintf('Re-converged in %d iterations\n',iter);
end

%% 6.  Economic Analysis and Diagnostics ----------------------------------
fprintf('\n=== Economic Diagnostics ===\n');

% Find steady states (where k' ≈ k)
[~, ss_idx_bad] = min(abs(Kpol(:,1) - k_grid));
[~, ss_idx_good] = min(abs(Kpol(:,2) - k_grid));

fprintf('\n--- Approximate Steady States ---\n');
fprintf('Bad shock (ε = %.3f):\n', eps_grid(1));
fprintf('  k = %.3f, k'' = %.3f, ratio k''/k = %.3f\n', ...
        k_grid(ss_idx_bad), Kpol(ss_idx_bad,1), Kpol(ss_idx_bad,1)/k_grid(ss_idx_bad));
fprintf('Good shock (ε = %.3f):\n', eps_grid(2));
fprintf('  k = %.3f, k'' = %.3f, ratio k''/k = %.3f\n', ...
        k_grid(ss_idx_good), Kpol(ss_idx_good,2), Kpol(ss_idx_good,2)/k_grid(ss_idx_good));

% Analyze depreciation and capacity utilization
fprintf('\n--- Depreciation Analysis (CORRECTED) ---\n');
for ie = 1:2
    % At steady state
    h = h_star(ss_idx_bad, ie);
    delta = B*h^omega/omega;          % CORRECTED formula
    fprintf('State ε = %.3f:\n', eps_grid(ie));
    fprintf('  Capacity utilization h = %.3f\n', h);
    fprintf('  Depreciation rate δ(h) = %.3f (%.1f%% per period)\n', delta, 100*delta);
    fprintf('  Survival rate 1-δ(h) = %.3f\n', 1-delta);
    fprintf('  Effective depreciation with shock = %.3f\n', delta*exp(-eps_grid(ie)));
    
    % At median capital
    h_med = h_star(round(Nk/2), ie);
    delta_med = B*h_med^omega/omega;  % CORRECTED
    l_med = l_star(round(Nk/2), ie);
    y_med = A*(k_grid(round(Nk/2))*h_med)^alpha * l_med^(1-alpha);
    fprintf('  At median k: y = %.3f, h = %.3f, l = %.3f, δ = %.3f\n', ...
            y_med, h_med, l_med, delta_med);
end

% Check investment constraints
fprintf('\n--- Investment Constraint Analysis ---\n');
n_constrained = zeros(2,1);
for ie = 1:2
    for ik = 1:Nk
        h = h_star(ik,ie);
        delta = B*h^omega/omega;      % CORRECTED
        min_kprime = k_grid(ik)*(1-delta)*exp(-eps_grid(ie));
        if abs(Kpol(ik,ie) - min_kprime) < 1e-6
            n_constrained(ie) = n_constrained(ie) + 1;
        end
    end
end
fprintf('States at investment constraint:\n');
fprintf('  Bad shock:  %d out of %d (%.1f%%)\n', ...
        n_constrained(1), Nk, 100*n_constrained(1)/Nk);
fprintf('  Good shock: %d out of %d (%.1f%%)\n', ...
        n_constrained(2), Nk, 100*n_constrained(2)/Nk);

% Check boundary behavior and grid artifacts
bound_bad = sum(Kpol(:,1) == k_grid(1));  % exactly at k_min
bound_good = sum(Kpol(:,2) == k_grid(1));

% Check for grid artifacts (vertical drops in policy function)
grid_artifact_bad = false;
grid_artifact_good = false;
for ik = 2:Nk-1
    % Check for sudden drops to k_min (more than 50% drop)
    if Kpol(ik,1) > 1.5*k_grid(1) && Kpol(ik+1,1) == k_grid(1)
        grid_artifact_bad = true;
    end
    if Kpol(ik,2) > 1.5*k_grid(1) && Kpol(ik+1,2) == k_grid(1)
        grid_artifact_good = true;
    end
end

if bound_bad > 0 || bound_good > 0 || grid_artifact_bad || grid_artifact_good
    fprintf('\n--- Boundary Behavior Analysis ---\n');
    if bound_bad > 0 || bound_good > 0
        fprintf('States at k_min:\n');
        fprintf('  Bad shock:  %d states\n', bound_bad);
        fprintf('  Good shock: %d states\n', bound_good);
    end
    
    if grid_artifact_bad || grid_artifact_good
        fprintf('\n*** VERTICAL DROP DETECTED ***\n');
        fprintf('The policy function drops vertically to k_min.\n');
        fprintf('This is a NUMERICAL ARTIFACT, not an economic optimum.\n');
        fprintf('\nCause: Investment non-negativity k'' >= k(1-δ)e^{-ε} forces\n');
        fprintf('       k'' to be at least ~95%% of k, creating infeasibility.\n');
        fprintf('\nWith k_min = %.1f%% of k_ss, this should not occur.\n', 100*k_min/k_ss);
        fprintf('If it persists, raise k_min to 25%% of k_ss.\n');
    end
end

%% 7.  Plots --------------------------------------------------------------
figure(1)
plot(k_grid,V(:,1),'b-','LineWidth',1.6); hold on
plot(k_grid,V(:,2),'r--','LineWidth',1.6);
xlabel('Capital k','FontSize',11); 
ylabel('Value function V','FontSize',11);
legend('\epsilon = -\Theta','\epsilon = +\Theta','Location','SouthEast');
title('Value function by shock state','FontSize',12); 
grid on;
set(gca,'FontSize',10);

figure(2)
plot(k_grid,Kpol(:,1),'b-','LineWidth',1.6); hold on
plot(k_grid,Kpol(:,2),'r--','LineWidth',1.6);
plot(k_grid,k_grid,'k:');          % 45° reference
xlabel('Current Capital k','FontSize',11); 
ylabel('Next-period capital k''','FontSize',11);
legend('\epsilon = -\Theta','\epsilon = +\Theta','45° line','Location','NorthWest');
title('Policy function K(k,\epsilon)','FontSize',12); 
grid on;
set(gca,'FontSize',10);

% Mark steady states
plot(k_grid(ss_idx_bad), Kpol(ss_idx_bad,1), 'bo', 'MarkerSize', 8);
plot(k_grid(ss_idx_good), Kpol(ss_idx_good,2), 'ro', 'MarkerSize', 8);

% Add informative text
xlim_curr = xlim;
ylim_curr = ylim;
text(xlim_curr(1) + 0.02*diff(xlim_curr), ylim_curr(2) - 0.05*diff(ylim_curr), ...
     'Depreciation: \delta(h) = Bh^\omega/\omega', ...
     'FontSize',9,'Color',[0.5 0.5 0.5],'Interpreter','tex');

%% 8.  Additional diagnostic plots ----------------------------------------
figure(3)
subplot(2,2,1)
plot(k_grid, h_star(:,1), 'b-', 'LineWidth', 1.5); hold on
plot(k_grid, h_star(:,2), 'r--', 'LineWidth', 1.5);
xlabel('Capital k'); ylabel('Capacity utilization h');
title('Optimal Capacity Utilization');
legend('\epsilon = -\Theta','\epsilon = +\Theta','Location','SouthEast');
grid on;

subplot(2,2,2)
delta_bad = B*h_star(:,1).^omega/omega;    % CORRECTED
delta_good = B*h_star(:,2).^omega/omega;   % CORRECTED
plot(k_grid, 100*delta_bad, 'b-', 'LineWidth', 1.5); hold on
plot(k_grid, 100*delta_good, 'r--', 'LineWidth', 1.5);
xlabel('Capital k'); ylabel('Depreciation rate \delta(h) (%)');
title('Endogenous Depreciation Rate');
legend('\epsilon = -\Theta','\epsilon = +\Theta','Location','SouthEast');
grid on;

subplot(2,2,3)
plot(k_grid, l_star(:,1), 'b-', 'LineWidth', 1.5); hold on
plot(k_grid, l_star(:,2), 'r--', 'LineWidth', 1.5);
xlabel('Capital k'); ylabel('Labor supply l');
title('Optimal Labor Supply');
legend('\epsilon = -\Theta','\epsilon = +\Theta','Location','SouthEast');
grid on;

subplot(2,2,4)
% Investment rates
inv_bad = (Kpol(:,1) - k_grid.*(1-delta_bad).*exp(-eps_grid(1))) ./ k_grid;
inv_good = (Kpol(:,2) - k_grid.*(1-delta_good).*exp(-eps_grid(2))) ./ k_grid;
plot(k_grid, 100*inv_bad, 'b-', 'LineWidth', 1.5); hold on
plot(k_grid, 100*inv_good, 'r--', 'LineWidth', 1.5);
xlabel('Capital k'); ylabel('Investment rate i/k (%)');
title('Investment Rate');
legend('\epsilon = -\Theta','\epsilon = +\Theta','Location','NorthEast');
grid on;
ylim([-50 50]);  % Reasonable bounds for investment rate

%% 9.  Save workspace (minimal) -------------------------------------------
save rbc_vfi_results.mat V Kpol k_grid eps_grid;
fprintf('\nMinimal results saved to rbc_vfi_results.mat\n');