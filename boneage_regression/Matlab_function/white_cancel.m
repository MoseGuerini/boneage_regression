function img_withoutL = white_cancel(img)
    img = im2double(img); % Converti in double per compatibilità
    T = 8 / 10; % Soglia abbassata per vedere l'effetto

    % Crea una maschera dei pixel sopra soglia
    mask1 = img > T;

    % Sostituisci solo i pixel sopra la soglia con la media dei vicini
    img_withoutL = img;
    img_withoutL(mask1) = 0;
    img2 = img_withoutL;

    D = 0.; % Soglia di differenza (0 = identico, 1 = massimo contrasto)
    
    % Crea un filtro 5x5 per calcolare la media dei vicini
    kernel = ones(30,30) / 900; % Media su tutta la finestra 3x3
    mean_neighbors = imfilter(img, kernel, 'replicate'); 

    kernel2 = ones(5,5) / 25; % Media su tutta la finestra 3x3
    mean_neighbors2 = imfilter(img, kernel2, 'replicate'); 

    % Calcola la differenza assoluta tra il pixel centrale e la media dei vicini
    diff = abs(img - mean_neighbors);

    % Crea una maschera dei pixel che sono troppo diversi dai vicini
    mask2 = diff > D;

    % Sostituisci solo i pixel molto diversi con 0
    for i = 1:20
        img_withoutL(mask2) = 0;
    end
    % for i = 1:2
    %     img_withoutL(mask2) = mean_neighbors2(mask2);
    % end
    
    % Mostra l'immagine originale e quella modificata
    figure;
    subplot(1,3,1); imshow(img); title('Immagine Originale');
    subplot(1,3,2); imshow(img2); title('Immagine senza bianchi estremi');
    subplot(1,3,3); imshow(img_withoutL); title('Immagine senza outliers');
end

% Ottieni il percorso completo dell'immagine
img_path = 'C:\Users\nicco\Desktop\Training_output\1377.png';

% Leggi l'immagine
img = imread(img_path);

% Processa l'immagine
img_modified = white_cancel(img); % CORRETTA chiamata della funzione

% Convertiamo entrambe le immagini in uint8 per garantire una visualizzazione coerente
img = im2uint8(img);
img_modified = im2uint8(img_modified);