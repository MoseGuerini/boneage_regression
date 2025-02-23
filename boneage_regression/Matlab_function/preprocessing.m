function preprocessing2(input_folder, output_folder,num_workers,target_size)

    % Se num_workers non è stato specificato, imposta il valore di default a 12
    
    if ~exist('num_workers','var')
         % third parameter does not exist, so default it to something
        num_workers = 12;
    end

    if ~exist('target_size','var')
         % third parameter does not exist, so default it to something
        target_size = 256;
    end

    % Aggiungi il percorso della cartella al MATLAB path per il parfor
    addpath(genpath(pwd));

    % Crea la cartella di output se non esiste
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end
    
    % Elimina i file esistenti nella cartella di output
    delete(fullfile(output_folder, '*.png'));

    % Ottiene la lista di tutti i file immagine nella cartella
    image_files = dir(fullfile(input_folder, '*.png')); 
    
    % Prealloca cell array per le immagini elaborate
    num_images = length(image_files);
    processed_images = cell(1, num_images);

    max_workers = parcluster('local').NumWorkers;
    num_workers = min(num_workers, max_workers); % Limita il numero di worker

    poolobj = gcp('nocreate'); 
    if isempty(poolobj) || poolobj.NumWorkers ~= num_workers
        delete(poolobj); % Chiude eventuali pool aperti con numero errato
        parpool(num_workers); % Avvia il pool con il numero corretto di worker
    end

    tic;  % Inizia il timer
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

        % Uscita in 128x128
        processed_images{i} = image_resizing(img_padded, target_size);

        % Salva l'immagine preprocessata nella cartella di output
        output_path = fullfile(output_folder, image_files(i).name);
        imwrite(processed_images{i}, output_path);
    end
    
    disp('Preprocessing completed. Preprocessed images: ', num2str(num_images)]);
    
    % Ferma il timer e mostra il tempo trascorso
    elapsed_time = toc;
    disp(['Elapsed time: ', num2str(elapsed_time), ' seconds']);
end