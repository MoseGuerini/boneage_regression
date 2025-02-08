function preprocessing(input_folder, output_folder)
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

        % Normalizza e ritaglia l'immagine
        img_cropped = image_cropping(img);

        % Padding e ridimensionamento dell'immagine
        img_padded = add_padding_to_square(img_cropped);

        %Come output da immagini 128x128
        processed_images{i} = image_resizing(img_padded, 128);

        % Salva l'immagine preprocessata nella cartella di output
        output_path = fullfile(output_folder, image_files(i).name);
        imwrite(processed_images{i}, output_path);  % Salva img_resized
    end
    
    disp('Elaborazione completata!');
end

preprocessing('C:\Users\nicco\Desktop\Training', 'C:\Users\nicco\Desktop\Preprocessed_foto')