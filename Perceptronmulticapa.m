% Cargar los datos
data = csvread('concentlite.csv');
inputs = data(:, 1:end-1)';
targets = data(:, end)';

hiddenLayerSize = [50, 50, 50]; % Ejemplo de 3 capas ocultas con 50 neuronas 

% Permitir al usuario seleccionar la regla de aprendizaje
fprintf('Seleccione la regla de aprendizaje:\n');
fprintf('1. Descenso de gradiente (traingd)\n');
fprintf('2. Descenso de gradiente con adaptación (traingda)\n');
fprintf('3. Descenso de gradiente con momento (traingdm)\n');
option = input('Opción: ');

switch option
    case 1
        % Crear la red neuronal
        net = patternnet(hiddenLayerSize,'trainscg', 'mse'); 
        
        % Dividir los datos
        net.divideParam.trainRatio = 0.7; % 70% de los datos para entrenamiento
        net.divideParam.valRatio = 0.1; % 10% de los datos para validación
        net.divideParam.testRatio = 0.2; % 20% de los datos para pruebas
        
        % Establecer el número de épocas
        net.trainParam.epochs = 1000;
        % Establecer la tasa de aprendizaje
        net.trainParam.lr = 0.01;
        net.trainFcn = 'traingd';
    case 2
        % Crear la red neuronal
        net = patternnet(hiddenLayerSize,'trainscg', 'mse'); 
        
        % Dividir los datos
        net.divideParam.trainRatio = 0.7; % 70% de los datos para entrenamiento
        net.divideParam.valRatio = 0.1; % 10% de los datos para validación
        net.divideParam.testRatio = 0.2; % 20% de los datos para pruebas
        
        % Establecer el número de épocas
        net.trainParam.epochs = 1000;
        % Establecer la tasa de aprendizaje
        net.trainParam.lr = 0.01;
        net.trainFcn = 'traingda';
    case 3
        % Crear la red neuronal
        net = patternnet(hiddenLayerSize,'trainscg', 'mse'); 
        
        % Dividir los datos
        net.divideParam.trainRatio = 0.7; % 70% de los datos para entrenamiento
        net.divideParam.valRatio = 0.1; % 10% de los datos para validación
        net.divideParam.testRatio = 0.2; % 20% de los datos para pruebas
        
        % Establecer el número de épocas
        net.trainParam.epochs = 1000;
        % Establecer la tasa de aprendizaje
        net.trainParam.lr = 0.01;
        net.trainFcn = 'traingdm';
    otherwise
        error('Opción no válida');
end

% Entrenar la red
net = train(net, inputs, targets);

% Simular la red
predictions = round(sim(net, inputs)); % Redondea las predicciones a -1 o 1

% Calcular la precisión
accuracy = sum(predictions == targets) / length(targets);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);

% Generar una cuadrícula de puntos en el espacio de entrada
[x1Grid, x2Grid] = meshgrid(linspace(min(inputs(1,:)), max(inputs(1,:)), 100), linspace(min(inputs(2,:)), max(inputs(2,:)), 100));
xGrid = [x1Grid(:)'; x2Grid(:)'];

% Utilizar la red para predecir la clase de cada punto en la cuadrícula
predictionsGrid = round(sim(net, xGrid));

% Visualizar las regiones de decisión
contourf(x1Grid, x2Grid, reshape(predictionsGrid, size(x1Grid)), 'LineStyle', 'none');
hold on;
scatter(inputs(1, targets == 1), inputs(2, targets == 1), 'ro'); % Puntos de la clase 1
scatter(inputs(1, targets == -1), inputs(2, targets == -1), 'bo'); % Puntos de la clase -1
hold off;
