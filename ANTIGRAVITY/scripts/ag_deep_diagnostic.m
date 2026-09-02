% Deep diagnostic of Moderate vs No_DR representation on V4.1, V4.2, and V4.3
clear; clc;
addpath('C:\Users\Abhij\Documents\DR_SIH\ANTIGRAVITY\scripts');

load('C:\Users\Abhij\Documents\DR_SIH\ANTIGRAVITY\checkpoints\V4_1_base.mat', 'valScoresV41', 'YValV41', 'classNamesV4');
load('C:\Users\Abhij\Documents\DR_SIH\ANTIGRAVITY\results\AG_V4_2_LR5e5_results.mat', 'valScores', 'YVal');
scoresV42 = valScores;
load('C:\Users\Abhij\Documents\DR_SIH\ANTIGRAVITY\results\AG_V4_3_LR2e5_results.mat', 'valScores');
scoresV43 = valScores;

% Moderate cases
modMask = (YVal == 'Moderate');
noDRMask = (YVal == 'No_DR');

fprintf('=== DEEP MODERATE DR ERROR DIAGNOSTICS (VALIDATION N=794 Moderate) ===\n\n');

for m = 1:3
    if m == 1
        name = 'V4.1 Baseline';
        sc = valScoresV41;
        th = 0.1460;
    elseif m == 2
        name = 'AG_V4_2_LR5e5';
        sc = scoresV42;
        th = 0.1413;
    else
        name = 'AG_V4_3_LR2e5';
        sc = scoresV43;
        th = 0.1290;
    end
    
    pMod = sc(modMask, 2);
    pNoDR = sc(modMask, 3);
    pRef = sum(sc(modMask, [2, 4, 5]), 2);
    
    % Clinical False Negatives for Moderate
    fnMod = (pRef < th);
    
    fprintf('--- %s ---\n', name);
    fprintf('Clinical FN on Moderate: %d / %d (%.2f%%)\n', sum(fnMod), sum(modMask), 100*mean(fnMod));
    fprintf('Moderate P(Mod):  mean=%.4f, median=%.4f, std=%.4f\n', mean(pMod), median(pMod), std(pMod));
    fprintf('Moderate P(NoDR): mean=%.4f, median=%.4f, std=%.4f\n', mean(pNoDR), median(pNoDR), std(pNoDR));
    
    % Confidence analysis of FN
    pNoDR_fn = pNoDR(fnMod);
    pMod_fn = pMod(fnMod);
    fprintf('FN cases breakdown:\n');
    fprintf('  High-confidence No_DR (P(NoDR) >= 0.80): %d / %d (%.1f%%)\n', ...
        sum(pNoDR_fn >= 0.80), sum(fnMod), 100*mean(pNoDR_fn >= 0.80));
    fprintf('  Moderate-confidence No_DR (0.60 <= P(NoDR) < 0.80): %d / %d (%.1f%%)\n', ...
        sum(pNoDR_fn >= 0.60 & pNoDR_fn < 0.80), sum(fnMod), 100*mean(pNoDR_fn >= 0.60 & pNoDR_fn < 0.80));
    fprintf('  Ambiguous / Low margin (P(NoDR) < 0.60): %d / %d (%.1f%%)\n', ...
        sum(pNoDR_fn < 0.60), sum(fnMod), 100*mean(pNoDR_fn < 0.60));
    fprintf('\n');
end

% Save figure of probability distributions
fig = figure('Visible', 'off', 'Position', [100 100 900 600]);
subplot(2, 2, 1);
histogram(valScoresV41(modMask, 2), 30, 'FaceColor', 'b', 'FaceAlpha', 0.5); hold on;
histogram(scoresV43(modMask, 2), 30, 'FaceColor', 'r', 'FaceAlpha', 0.5);
title('P(Moderate) on True Moderate'); xlabel('Probability'); ylabel('Count');
legend('V4.1 Base', 'AG V4.3 LR2e5');

subplot(2, 2, 2);
histogram(valScoresV41(modMask, 3), 30, 'FaceColor', 'b', 'FaceAlpha', 0.5); hold on;
histogram(scoresV43(modMask, 3), 30, 'FaceColor', 'r', 'FaceAlpha', 0.5);
title('P(No DR) on True Moderate'); xlabel('Probability'); ylabel('Count');
legend('V4.1 Base', 'AG V4.3 LR2e5');

subplot(2, 2, 3);
histogram(valScoresV41(noDRMask, 2), 30, 'FaceColor', 'b', 'FaceAlpha', 0.5); hold on;
histogram(scoresV43(noDRMask, 2), 30, 'FaceColor', 'r', 'FaceAlpha', 0.5);
title('P(Moderate) on True No DR'); xlabel('Probability'); ylabel('Count');
legend('V4.1 Base', 'AG V4.3 LR2e5');

subplot(2, 2, 4);
marginBase = valScoresV41(modMask, 2) - valScoresV41(modMask, 3);
marginV43 = scoresV43(modMask, 2) - scoresV43(modMask, 3);
histogram(marginBase, 30, 'FaceColor', 'b', 'FaceAlpha', 0.5); hold on;
histogram(marginV43, 30, 'FaceColor', 'r', 'FaceAlpha', 0.5);
title('Margin P(Mod) - P(No DR) on True Moderate'); xlabel('Margin'); ylabel('Count');
legend('V4.1 Base', 'AG V4.3 LR2e5');

saveas(fig, 'C:\Users\Abhij\Documents\DR_SIH\ANTIGRAVITY\figures\moderate_nodr_distribution.png');
fprintf('Saved figure to ANTIGRAVITY\\figures\\moderate_nodr_distribution.png\n');
