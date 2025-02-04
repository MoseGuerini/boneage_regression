function matlab_images_preprocessing(input_folder, output_folder)
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
        if size(img,3) == 3
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

        % Salva l'immagine elaborata nella cell array
        processed_images{i} = img_cropped;

        % Salva l'immagine preprocessata nella cartella di output
        output_path = fullfile(output_folder, image_files(i).name);
        imwrite(img_cropped, output_path);
    end

    disp('Elaborazione completata!');

end
