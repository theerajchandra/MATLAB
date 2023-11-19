function regression()
    % Main function for regression analysis.
    % Loads data from two text files and performs the selected regression analysis on both datasets.

    % Load data from text files
    data1 = load('test1.txt');
    data2 = load('test2.txt');

    % Combine data into a cell array for easier manipulation
    data = {data1, data2};

    % Display options for regression type to user
    fprintf('Select the function to fit your data:\n');
    fprintf('1. Polynomial: y=a0+a1x+...+amx^m\n');
    fprintf('2. Exponential: y=ae^bx\n');
    fprintf('3. Saturation: y=(ax/b+/Users/theerajchandra/Downloads/A3.mx)\n');

    % Get user input for regression type only once
    option = input('Enter the number of the selected function: ');

    % If polynomial regression is chosen, ask for the degree once
    degree = 0;
    if option == 1
        degree = input('Enter the degree of the polynomial: ');
    end

    % Loop through each dataset and apply the chosen regression
    for k = 1:2
        fprintf('\nPerforming analysis on test%d data...\n', k);
        switch option
            case 1
                [coefficients, formula, rSquared] = polynomial_regression(data{k}, degree);
            case 2
                [coefficients, formula, rSquared] = exponential_regression(data{k});
            case 3
                [coefficients, formula, rSquared] = saturation_regression(data{k});
        end

        % Plot and display the results
        plotAndDisplayResults(data{k}, coefficients, formula, rSquared, option);
    end
end

function performPolynomialRegression(data)
    % Handles polynomial regression.
    % Prompts for the degree of polynomial and performs regression on the provided data.

    degree = input('Enter the degree of the polynomial: ');

    [coefficients, formula, rSquared] = polynomial_regression(data, degree);

    plotAndDisplayResults(data, coefficients, formula, rSquared, 1);
end

function performExponentialRegression(data)
    % Handles exponential regression.
    % Performs regression on the provided data using the exponential model.

    [coefficients, formula, rSquared] = exponential_regression(data);

    plotAndDisplayResults(data, coefficients, formula, rSquared, 2);
end

function performSaturationRegression(data)
    % Handles saturation regression.
    % Performs regression on the provided data using the saturation model.

    [a, b, rSquared] = saturationFit(data(:,1), data(:,2));

    % Display the results or store them in variables if needed
    coefficients = [a, b];
    formula = sprintf('y = (%.6f * x) / (%.6f + x)', a, b);

    plotAndDisplayResults(data, coefficients, formula, rSquared, 3);
end

function plotAndDisplayResults(data, coefficients, formula, rSquared, option)
    % Plots the data and the regression result.
    % Displays the estimated formula, R-squared value, and plots the regression curve against the data.

    fprintf('\nResults:\n');
    fprintf('Estimated Formula: %s\n', formula);
    fprintf('R-squared: %f\n\n', rSquared); % Empty line added after R-squared value

    figure;
    scatter(data(:,1), data(:,2), 'b', 'DisplayName', 'Raw Data');
    hold on;
    x_vals = linspace(min(data(:,1)), max(data(:,1)), 100);
    y_vals = calculate_regression(x_vals, coefficients, option);
    plot(x_vals, y_vals, 'r', 'DisplayName', 'Estimated Function');
    legend('Location', 'Best');
    title('Regression Results');
    xlabel('x');
    ylabel('y');
    text(0.05, 0.9, formula, 'Units', 'normalized', 'FontSize', 10, 'Color', 'k');
    text(0.05, 0.85, ['R^2 = ' num2str(rSquared, '%.6f')], 'Units', 'normalized', 'FontSize', 10, 'Color', 'k');
end

function y_vals = calculate_regression(x_vals, coefficients, option)
    % Calculates the y-values based on the selected regression type.
    % Uses the coefficients and the type of regression to compute the y-values for the given x-values.

    switch option
        case 1 % Polynomial regression
            y_vals = evaluate_polynomial(coefficients, x_vals);
        case 2 % Exponential regression
            y_vals = exp(coefficients(1) + coefficients(2) * x_vals);
        case 3 % Saturation regression
            a = coefficients(1);
            b = coefficients(2);
            y_vals = (a * x_vals) ./ (b + x_vals);
    end
end

function y = evaluate_polynomial(coefficients, x)
    % Evaluates a polynomial for a given set of coefficients and x-values.
    % The coefficients are in descending order of power.

    n = length(coefficients);
    y = zeros(size(x));
    for i = 1:n
        y = y + coefficients(i) * x.^(n - i);
    end
end

function [coefficients, formula, rSquared] = polynomial_regression(data, degree)
    % Performs polynomial regression on the provided dataset.
    % Fits a polynomial of the specified degree to the data.

    x = data(:,1);
    y = data(:,2);

    % Construct Vandermonde matrix for the polynomial fit
    A = zeros(length(x), degree + 1);
    for i = 1:degree + 1
        A(:, i) = x.^(degree + 1 - i);
    end

    % Solve linear system to find coefficients
    coefficients = A\y;

    % Generate formula string for the polynomial
    formula = generate_polynomial_formula(coefficients);

    % Calculate R-squared for the fit
    y_fit = evaluate_polynomial(coefficients, x);
    rSquared = calculate_r_squared(y, y_fit);
end

function formula = generate_polynomial_formula(coefficients)
    % Generates a string representation of a polynomial formula.
    % The coefficients are presented in a descending order of power.

    degree = length(coefficients) - 1;
    formula = 'y = ';
    for i = 1:degree
        formula = [formula sprintf('%.4fx^%d + ', coefficients(i), degree + 1 - i)];
    end
    formula = [formula sprintf('%.4f', coefficients(end))];
end

function [coefficients, formula, rSquared] = exponential_regression(data)
    % Performs exponential regression on the provided dataset.
    % Fits an exponential model to the data.

    x = data(:,1);
    y = data(:,2);

    % Linearize the data for the exponential fit
    A = [ones(length(x), 1), x];
    b = log(y);

    % Solve for coefficients of the linearized form
    coefficients = A\b;

    % Convert coefficients back to exponential form
    formula = ['y = ' num2str(exp(coefficients(1)), '%.4f') 'e^{' num2str(coefficients(2), '%.4f') 'x}'];

    % Calculate R-squared for the exponential fit
    y_fit = exp(coefficients(1) + coefficients(2) * x);
    rSquared = calculate_r_squared(y, y_fit);
end
function [coefficients, formula, rSquared] = saturation_regression(data)
    % Performs saturation regression on the provided dataset.
    % Fits a saturation model to the data and returns the coefficients, formula, and R-squared value.

    x = data(:,1);
    y = data(:,2);

    % Fit the data using saturation model
    [a, b, rSquared] = saturationFit(x, y);

    % Formulate the coefficients and formula for saturation regression
    coefficients = [a, b];
    formula = sprintf('y = (%.6f * x) / (%.6f + x)', a, b);
end

function [a, b, rSquared] = saturationFit(x, y)
    % Performs saturation regression on the provided dataset.
    % Fits a saturation model to the data.

    % Ensure x and y are column vectors
    x = x(:);
    y = y(:);

    % Linearize the model for fitting
    Y = 1 ./ y;
    X = 1 ./ x;

    % Perform linear regression on transformed data
    n = length(X);
    sumX = sum(X);
    sumY = sum(Y);
    sumXY = sum(X .* Y);
    sumXX = sum(X .* X);

    % Calculate slope (m) and intercept (C) of the linearized form
    m = (n * sumXY - sumX * sumY) / (n * sumXX - sumX^2);
    C = (sumY - m * sumX) / n;

    % Convert linear model coefficients to saturation model parameters
    a = 1 / C;
    b = m / C;

    % Calculate R-squared for the saturation fit
    yFit = (a * x) ./ (b + x);
    rSquared = calculate_r_squared(y, yFit);
end

function r_squared = calculate_r_squared(y_actual, y_predicted)
    % Calculates the R-squared value to assess the fit quality.
    % R-squared is a statistical measure of how close the data are to the fitted regression line.

    SS_residual = sum((y_actual - y_predicted).^2);
    SS_total = sum((y_actual - mean(y_actual)).^2);
    r_squared = 1 - (SS_residual / SS_total);
end
