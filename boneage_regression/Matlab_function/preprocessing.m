function preprocessing2(input_folder, output_folder, num_workers, target_size)

    % If num_workers has not been specified, set 12
    
    if ~exist('num_workers','var')
        % if parameter does not exist, so default it to something
        num_workers = 12;
    end

    if ~exist('target_size','var')
        % if parameter does not exist, so default it to something
        target_size = 256;
    end

    % Adding folder path to MATLAB path
    addpath(genpath(pwd));

    % Create output folder if it does not exist
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end
    
    % Clean output folder if it exists
    delete(fullfile(output_folder, '*.png'));

    % Obtain png list of the input folder
    image_files = dir(fullfile(input_folder, '*.png')); 
    
    % Preallocate space for preprocessed images
    num_images = length(image_files);
    processed_images = cell(1, num_images);

    max_workers = parcluster('local').NumWorkers;
    % Chose the minumum between workers required and workers available
    num_workers = min(num_workers, max_workers);

    poolobj = gcp('nocreate'); 
    if isempty(poolobj) || poolobj.NumWorkers ~= num_workers
        delete(poolobj); % If there are pools using the wrong number of
        % workers, close it
        parpool(num_workers); % Start pool with right number of workers
    end

    tic; % Useful if you are interested in how much time does the
    % preprocessing need
    % Start parallel loop over the images
    parfor i = 1:num_images

        % Obtaining image path
        img_path = fullfile(input_folder, image_files(i).name);

        % Read image
        img = imread(img_path);

        % Normalize and crop the image
        img_cropped = image_cropping(img);

        % Pad and resize the image
        img_padded = add_padding_to_square(img_cropped);

        % Resize the image
        processed_images{i} = image_resizing(img_padded, target_size);

        % Save the image in the output folder
        output_path = fullfile(output_folder, image_files(i).name);
        imwrite(processed_images{i}, output_path);
    end
    
    disp( ...
    ['Preprocessing completed. Preprocessed images: ', ...
    num2str(num_images)] ...
    );
    
    % Stop the timer and visualize how much time preprocessing needed
    elapsed_time = toc;
    disp(['Elapsed time: ', num2str(elapsed_time), ' seconds']);
end