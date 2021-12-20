load data.mat

C1_train = classes(1:80,:);
C1_test = classes(81:160,:);

C2_train = classes(161:240,:);
C2_test = classes(241:320,:);

C3_train = classes(321:400,:);
C3_test = classes(401:480,:);

C4_train = classes(481:560,:);
C4_test = classes(561:640,:);

train_set = {C1_train, C2_train, C3_train, C4_train};
test_set = {C1_test, C2_test, C3_test, C4_test};

sigmas = {eye(8), eye(8), eye(8), eye(8)};
means = {(sum(C1_train)/80)', (sum(C2_train)/80)', (sum(C3_train)/80)', (sum(C4_train)/80)'};

disp("Beginning: ")
disp("  Feature Set:")
disp(1:8)
disp("  Classification Accuracy:")
disp(FindAccuracy(means, sigmas, test_set, 1:8))

sfs_X = sfs(means, sigmas, test_set, 6);
disp("SFS: ")
disp("  Optimum Feature Subset:")
disp(sfs_X)
disp("  Classification Accuracy:")
disp(FindAccuracy(means, sigmas, test_set, sfs_X))

sbs_X = sbs(means, sigmas, test_set, 6);
disp("SBS: ")
disp("  Optimum Feature Subset:")
disp(sbs_X)
disp("  Classification Accuracy:")
disp(FindAccuracy(means, sigmas, test_set, sbs_X))

%function [sgm, m] = DefineSigmaAndMean(x)
%    sgm = zeros(size(x,2), size(x,2));
%    m = (sum(x)/length(x))';
%    
%    for f_vector = x'
%        sgm = sgm + (f_vector - m) * (f_vector - m)';
%    end
%end

function acc = FindAccuracy(means, sigmas, test_set, test_x)
    truth_num = 0;
            for c = 1:4
                for vector_no = 1:80
                    vector = zeros(1,8);
                    vector(test_x) = 1;
                    vector = vector .* test_set{c}(vector_no, :);
                    predicted_class = BayesianClassifier(means, sigmas, vector');
                    if predicted_class == c
                        truth_num = truth_num + 1;
                    end
                end
            end
    acc = truth_num * 100 / 320;
end

function X = sfs(means, sigmas, test_set, p)
    X = [];
    x = 1:8;

    while p ~= length(X)
        highest_acc = 0;
        highest_class = 0;
        for i = x
            test_x = [X,i];
            acc = FindAccuracy(means, sigmas, test_set, test_x);
            if acc > highest_acc
                highest_acc = acc;
                highest_class = i;
            end
        end
        X = [X, highest_class]; %#ok<AGROW> 
        x = x(highest_class ~= x);
    end
end

function X = sbs(means, sigmas, test_set, p)
    X = 1:8;

    while p ~= length(X)
        highest_acc = 0;
        highest_class = 0;
        for i = X
            test_x = X(i ~= X);
            acc = FindAccuracy(means, sigmas, test_set, test_x);
            if acc > highest_acc
                highest_acc = acc;
                highest_class = i;
            end
        end
        X = X(highest_class ~= X);
    end
end

function class = BayesianClassifier(means, sigmas, x)
    biggest = 0;
    for i = 1:4
        bayes = 1/4 * comp_gauss_dens_val(means{i}, sigmas{i}, x);
        if bayes > biggest
            biggest = bayes;
            class = i;
        end
    end
end

function pg = comp_gauss_dens_val(m, S, x)
    pg = exp(-(x-m)' * (inv(S)) * (x-m) / 2) / sqrt((2*pi)^(size(x, 1)) * det(S));
end
