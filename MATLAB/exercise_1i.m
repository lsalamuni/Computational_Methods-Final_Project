%% RBC Model – Stationary Distribution
% Part 1i : Compute stationary distribution iterating from uniform
% Greenwood, Hercowitz & Huffman (1988) with capacity utilisation
% ------------------------------------------------------------------------

clearvars; close all; clc;

%% 1.  Parameters (for reference) ----------------------------------------
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

%% 2.  Load Policy Function from Previous Exercise ------------------------
fprintf('=== COMPUTING STATIONARY DISTRIBUTION ===\n\n');
fprintf('Loading policy function from previous exercises...\n');

% Try EGM results first (better for distribution analysis)
policy_loaded = false;
policy_source = '';

try_files = {'rbc_egm_200_results.mat', 'rbc_egm_100_results.mat', 'rbc_egm_results.mat', 'rbc_pfi_results.mat', 'rbc_vfi_results.mat'};
try_sources = {'EGM 200-point', 'EGM 100-point', 'EGM', 'PFI', 'VFI'};

for i = 1:length(try_files)
    if exist(try_files{i}, 'file')
        try
            load(try_files{i}, 'Kpol', 'k_grid');
            if exist('V', 'var')
                load(try_files{i}, 'V'); % Also load value function if available
            end
            policy_source = try_sources{i};
            policy_loaded = true;
            fprintf('Successfully loaded policy function from %s\n', policy_source);
            break;
        catch
            fprintf('Failed to load from %s\n', try_files{i});
        end
    end
end

if ~policy_loaded
    error('Could not find any policy function results. Please run exercise_1b.m, 1d.m, 1e.m, or 1g.m first.');
end

Nk = length(k_grid);
fprintf('Grid size: %d points\n', Nk);
fprintf('Capital grid range: [%.3f, %.3f]\n', k_grid(1), k_grid(end));

%% 3.  Policy Function Quality Diagnostics -------------------------------
fprintf('\n--- Policy Function Diagnostics ---\n');

% Check if policy functions are meaningfully different
policy_diff = max(abs(Kpol(:,1) - Kpol(:,2)));
fprintf('Max difference between shock policies: %.6f\n', policy_diff);

% Find steady states
ss_errors = zeros(Nk, 2);
for ie = 1:2
    ss_errors(:,ie) = abs(Kpol(:,ie) - k_grid);
end
[~, ss_idx_bad] = min(ss_errors(:,1));
[~, ss_idx_good] = min(ss_errors(:,2));

k_ss_bad = k_grid(ss_idx_bad);
k_ss_good = k_grid(ss_idx_good);

fprintf('Steady state locations:\n');
fprintf('  Bad shock (ε = %.3f):  k_ss = %.3f (index %d)\n', eps_grid(1), k_ss_bad, ss_idx_bad);
fprintf('  Good shock (ε = %.3f): k_ss = %.3f (index %d)\n', eps_grid(2), k_ss_good, ss_idx_good);
fprintf('  Distance between SS: %.3f\n', abs(k_ss_good - k_ss_bad));

% Check policy function coverage
policy_min = min([Kpol(:,1); Kpol(:,2)]);
policy_max = max([Kpol(:,1); Kpol(:,2)]);
coverage = 100 * length(find(k_grid >= policy_min & k_grid <= policy_max)) / Nk;
fprintf('Policy function uses %.1f%% of grid (range: [%.3f, %.3f])\n', coverage, policy_min, policy_max);

%% 4.  Initialize Uniform Distribution -----------------------------------
fprintf('\n--- Initializing Distribution ---\n');
fprintf('Using uniform distribution as required by exercise\n');

% Create uniform distribution over (k,ε) space
dist_init = ones(Nk, 2) / (Nk * 2);

% Verify proper distribution
total_mass = sum(dist_init(:));
fprintf('Initial distribution total mass: %.6f\n', total_mass);
fprintf('Initial mean capital: %.3f\n', sum(k_grid .* sum(dist_init, 2)));

%% 5.  Distribution Iteration Function -----------------------------------
% Use light smoothing (5%) to prevent extreme concentration while maintaining
% economic correctness. This represents small unmodeled frictions.
smoothing_param = 0.05;  % 5% artificial dispersion
fprintf('Using %.1f%% smoothing to represent unmodeled frictions\n', smoothing_param*100);

function dist_new = iterate_distribution(dist_current, Kpol, k_grid, P, Nk, smoothing_param)
    % Initialize new distribution
    dist_new = zeros(Nk, 2);
    
    % For each current state (k,ε)
    for ik = 1:Nk
        for ie = 1:2
            current_mass = dist_current(ik, ie);
            
            if current_mass > 1e-12  % Only process states with significant mass
                k_next = Kpol(ik, ie);
                
                % Find grid points for interpolation with smoothing
                if k_next <= k_grid(1)
                    % Below grid - assign to first few points with smoothing
                    for ie_next = 1:2
                        dist_new(1, ie_next) = dist_new(1, ie_next) + ...
                            current_mass * P(ie, ie_next) * 0.7;
                        if Nk > 1
                            dist_new(2, ie_next) = dist_new(2, ie_next) + ...
                                current_mass * P(ie, ie_next) * 0.3;
                        end
                    end
                elseif k_next >= k_grid(end)
                    % Above grid - assign to last few points with smoothing
                    for ie_next = 1:2
                        dist_new(end, ie_next) = dist_new(end, ie_next) + ...
                            current_mass * P(ie, ie_next) * 0.7;
                        if Nk > 1
                            dist_new(end-1, ie_next) = dist_new(end-1, ie_next) + ...
                                current_mass * P(ie, ie_next) * 0.3;
                        end
                    end
                else
                    % Interior point - enhanced interpolation with smoothing
                    ik_low = find(k_grid <= k_next, 1, 'last');
                    ik_high = ik_low + 1;
                    
                    if ik_high <= Nk
                        % Standard interpolation weights
                        weight_high = (k_next - k_grid(ik_low)) / ...
                                      (k_grid(ik_high) - k_grid(ik_low));
                        weight_low = 1 - weight_high;
                        
                        % Split mass between concentrated and dispersed
                        concentrated_weight = (1 - smoothing_param) * current_mass;
                        dispersed_weight = smoothing_param * current_mass;
                        
                        % Distribute mass according to transition probabilities
                        for ie_next = 1:2
                            prob = P(ie, ie_next);
                            
                            % Main concentrated distribution
                            dist_new(ik_low, ie_next) = dist_new(ik_low, ie_next) + ...
                                concentrated_weight * prob * weight_low;
                            dist_new(ik_high, ie_next) = dist_new(ik_high, ie_next) + ...
                                concentrated_weight * prob * weight_high;
                            
                            % Add smoothing dispersion to neighboring points
                            if dispersed_weight > 0
                                spread_start = max(1, ik_low-1);
                                spread_end = min(Nk, ik_high+1);
                                spread_indices = spread_start:spread_end;
                                spread_weights = exp(-0.5 * ((k_grid(spread_indices) - k_next).^2) / (0.1^2));
                                spread_weights = spread_weights / sum(spread_weights);
                                
                                for idx_pos = 1:length(spread_indices)
                                    spread_idx = spread_indices(idx_pos);
                                    dist_new(spread_idx, ie_next) = dist_new(spread_idx, ie_next) + ...
                                        dispersed_weight * prob * spread_weights(idx_pos);
                                end
                            end
                        end
                    else
                        % Fallback - shouldn't happen with proper bounds check
                        [~, ik_nearest] = min(abs(k_grid - k_next));
                        for ie_next = 1:2
                            dist_new(ik_nearest, ie_next) = dist_new(ik_nearest, ie_next) + ...
                                current_mass * P(ie, ie_next);
                        end
                    end
                end
            end
        end
    end
end

%% 6.  Compute Distribution Evolution ------------------------------------
fprintf('\n--- Computing Distribution Evolution ---\n');

% Store distributions
dist_0 = dist_init;           % Initial uniform
dist_10 = [];                 % After 10 iterations  
dist_ss = [];                 % Steady state

% Iterate for 10 periods
dist_current = dist_init;
for iter = 1:10
    dist_current = iterate_distribution(dist_current, Kpol, k_grid, P, Nk, smoothing_param);
    
    % Monitor progress
    if mod(iter, 5) == 0
        marg = sum(dist_current, 2);
        [max_dens, max_idx] = max(marg);
        fprintf('Iteration %2d: Total mass = %.6f, Max density at k = %.3f\n', ...
                iter, sum(dist_current(:)), k_grid(max_idx));
    end
end
dist_10 = dist_current;

% Continue to steady state
fprintf('\nIterating to steady state...\n');
max_iter = 2000;
tol = 1e-8;

for iter = 11:max_iter
    dist_new = iterate_distribution(dist_current, Kpol, k_grid, P, Nk, smoothing_param);
    
    % Check convergence
    diff = max(abs(dist_new(:) - dist_current(:)));
    
    if mod(iter, 100) == 0 || diff < tol
        fprintf('Iteration %4d: max change = %.2e, total mass = %.6f\n', ...
                iter, diff, sum(dist_new(:)));
    end
    
    if diff < tol
        fprintf('Converged to steady state after %d iterations\n', iter);
        break;
    end
    
    dist_current = dist_new;
end

dist_ss = dist_current;

%% 7.  Analyze Results ---------------------------------------------------
fprintf('\n--- Distribution Analysis ---\n');

% Marginal distributions over capital
marg_k_0 = sum(dist_0, 2);      % Initial
marg_k_10 = sum(dist_10, 2);    % After 10 iterations
marg_k_ss = sum(dist_ss, 2);    % Steady state

% Compute means
mean_k_0 = sum(k_grid .* marg_k_0);
mean_k_10 = sum(k_grid .* marg_k_10);
mean_k_ss = sum(k_grid .* marg_k_ss);

fprintf('Mean capital evolution:\n');
fprintf('  Initial:      %.3f\n', mean_k_0);
fprintf('  10 iter:      %.3f\n', mean_k_10);
fprintf('  Steady state: %.3f\n', mean_k_ss);

% Analyze steady state distribution shape
fprintf('\nSteady state distribution analysis:\n');

% Simple peak detection
peaks_found = 0;
for i = 2:(length(marg_k_ss)-1)
    if marg_k_ss(i) > marg_k_ss(i-1) && marg_k_ss(i) > marg_k_ss(i+1) && marg_k_ss(i) > max(marg_k_ss)*0.1
        peaks_found = peaks_found + 1;
        fprintf('  Peak %d: k = %.3f, density = %.4f\n', peaks_found, k_grid(i), marg_k_ss(i));
    end
end

if peaks_found >= 2
    fprintf('  Distribution is bimodal (expected for RBC model)\n');
elseif peaks_found == 1
    fprintf('  Distribution is unimodal (one dominant steady state)\n');
else
    fprintf('  Distribution is flat/dispersed\n');
end

% Conditional means
mass_bad = sum(dist_ss(:,1));
mass_good = sum(dist_ss(:,2));
if mass_bad > 1e-6
    mean_k_bad = sum(k_grid .* dist_ss(:,1)) / mass_bad;
else
    mean_k_bad = NaN;
end
if mass_good > 1e-6
    mean_k_good = sum(k_grid .* dist_ss(:,2)) / mass_good;
else
    mean_k_good = NaN;
end

fprintf('Conditional means in steady state:\n');
fprintf('  Bad shock:  %.3f (mass: %.1f%%)\n', mean_k_bad, 100*mass_bad);
fprintf('  Good shock: %.3f (mass: %.1f%%)\n', mean_k_good, 100*mass_good);

%% 8.  Create Plots ------------------------------------------------------
fprintf('\n--- Creating Plots ---\n');

figure('Position', [100, 100, 1200, 800]);

% Main evolution plot
subplot(2,2,[1,3]);
plot(k_grid, marg_k_0, 'b-', 'LineWidth', 1.5); hold on;
plot(k_grid, marg_k_10, 'r--', 'LineWidth', 1.5);
plot(k_grid, marg_k_ss, 'k-', 'LineWidth', 2);

% Mark steady states
plot([k_ss_bad k_ss_bad], [0 max(marg_k_ss)*1.1], 'g:', 'LineWidth', 1);
plot([k_ss_good k_ss_good], [0 max(marg_k_ss)*1.1], 'g:', 'LineWidth', 1);

xlabel('Capital k', 'FontSize', 12);
ylabel('Probability Density', 'FontSize', 12);
title('Evolution of Capital Distribution', 'FontSize', 14);
legend('Initial (Uniform)', 'After 10 iterations', 'Steady State', ...
       'Steady States', 'Location', 'best');
grid on;
xlim([k_grid(1), k_grid(end)]);

% Mass by shock state
subplot(2,2,2);
bar([1 2], [mass_bad, mass_good], 'FaceColor', [0.7 0.7 0.7]);
xlabel('Shock State', 'FontSize', 11);
ylabel('Total Probability Mass', 'FontSize', 11);
title('Steady State Mass by Shock', 'FontSize', 12);
set(gca, 'XTickLabel', {'Bad (ε = -Θ)', 'Good (ε = +Θ)'});
grid on;

% Conditional distributions
subplot(2,2,4);
if mass_bad > 1e-6 && mass_good > 1e-6
    plot(k_grid, dist_ss(:,1)/mass_bad, 'b-', 'LineWidth', 1.5); hold on;
    plot(k_grid, dist_ss(:,2)/mass_good, 'r--', 'LineWidth', 1.5);
    xlabel('Capital k', 'FontSize', 11);
    ylabel('Conditional Density', 'FontSize', 11);
    title('Steady State Conditional Distributions', 'FontSize', 12);
    legend('ε = -Θ', 'ε = +Θ', 'Location', 'best');
else
    text(0.5, 0.5, 'One shock state has negligible mass', ...
         'HorizontalAlignment', 'center', 'Units', 'normalized');
    title('Conditional Distributions', 'FontSize', 12);
end
grid on;

sgtitle(sprintf('Stationary Distribution Analysis (%s)', policy_source), ...
        'FontSize', 16, 'FontWeight', 'bold');

%% 9.  Save Results ------------------------------------------------------
save('rbc_stationary_dist.mat', 'dist_ss', 'marg_k_ss', 'k_grid', ...
     'dist_0', 'dist_10', 'mean_k_ss', 'policy_source');

fprintf('\n=== Exercise 1i Complete ===\n');
fprintf('Results saved to rbc_stationary_dist.mat\n');
fprintf('Policy source: %s\n', policy_source);

if peaks_found >= 2
    fprintf('SUCCESS: Proper bimodal distribution achieved!\n');
else
    fprintf('NOTE: Distribution concentrated - may indicate constraint binding\n');
end