function matlab_images_preprocessing(input_folder, output_folder)
    % Aggiungi il percorso della cartella al MATLAB path per il parfor
    addpath(genpath(pwd));  % Aggiungi il percorso della cartella corrente se necessario

    % Crea la cartella di output se non esiste
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end
    
    % Elimina i file esistenti nella cartella di output
    delete(fullfile(output_folder, '*.png'));

    % Ottiene la lista di tutti i file immagine nella cartella
    image_files = dir(fullfile(input_folder, '*.png')); % Cambia l'estensione se necessario
    
    % Prealloca cell array per le immagini elaborate
    num_images = length(image_files);
    processed_images = cell(1, num_images);

    % Avvia il parallel pool se non è attivo
    poolobj = gcp('nocreate');
    if isempty(poolobj)
        parpool; % Avvia un pool di worker
    end

    % Loop parallelo su tutte le immagini
    parfor i = 1:num_images
        disp(['Processing image: ', num2str(i)]);

        % Ottieni il percorso completo dell'immagine
        img_path = fullfile(input_folder, image_files(i).name);

        % Leggi l'immagine
        img = imread(img_path);

        % Converti in grayscale se necessario
        if size(img, 3) == 3
            img = rgb2gray(img);
        end

        img = double(img); % Converti in double per compatibilità con Python

        % Normalizza l'immagine
        max_intensity = prctile(img(:), 99.5);
        min_intensity = prctile(img(:), 0.5);
        img_norm = (img - min_intensity) / (max_intensity - min_intensity);
        img_norm = max(0, min(1, img_norm));

        % Crea la maschera per trovare la mano
        lower_threshold = 0.1;
        upper_threshold = 1;
        mask = (img_norm > lower_threshold) & (img_norm < upper_threshold);

        % Trova il bounding box della mano
        [row, col] = find(mask);
        if isempty(row) || isempty(col)
            processed_images{i} = []; % Se non trova nulla, salva un array vuoto
            continue;
        end
        x_min = min(col);
        x_max = max(col);
        y_min = min(row);
        y_max = max(row);
        bbox = [x_min, y_min, x_max - x_min, y_max - y_min];

        % Ritaglia l'immagine
        img_cropped = imcrop(img_norm, bbox);

        %img_squared = add_padding_to_square2(img_cropped)

        % Salva l'immagine preprocessata nella cartella di output
        output_path = fullfile(output_folder, image_files(i).name);
        imwrite(img_cropped, output_path);  % Salva img_resized
    end

    disp('Elaborazione completata!');
end

function add_padding_to_square2(img)

    % Stampa la dimensione dell'immagine di partenza
    %disp(['Dimensione dell''immagine di partenza: ', mat2str(size(img))]);

    % Ottieni le dimensioni dell'immagine
    [height, width, channels] = size(img);

    % Calcola la dimensione del quadrato
    new_size = max(height, width);

    % Crea un'immagine quadrata nera (tutta 0)
    img_square = zeros(new_size, new_size, channels, 'like', img);

    % Calcola la posizione di partenza per centrare l'immagine originale
    row_start = floor((new_size - height) / 2) + 1;
    col_start = floor((new_size - width) / 2) + 1;

    % Copia l'immagine originale nel centro della nuova immagine quadrata
    img_square(row_start:row_start + height - 1, col_start:col_start + width - 1, :) = img;

    % Visualizza l'immagine quadrata
    figure;
    imshow(img_square);

    % Stampa la dimensione dell'immagine quadrata
    disp(['Dimensione dell''immagine quadrata: ', mat2str(size(img_square))]);

    % Salva l'immagine modificata (se desiderato)
    %imwrite(img_square, ['squared_' img_path]); % Per salvare l'immagine con il prefisso "squared_"
end

% Chiamata alla funzione
matlab_images_preprocessing('C:\Users\nicco\Desktop\foto', 'C:\Users\nicco\Desktop\Preprocessed_foto');
