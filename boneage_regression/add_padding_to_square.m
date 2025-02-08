function img_padded = add_padding_to_square(img)
    % Ottieni le dimensioni dell'immagine
    [height, width, channels] = size(img);

    % Calcola la dimensione del quadrato
    new_size = max(height, width);

    % Crea un'immagine quadrata nera (tutta 0)
    img_padded = zeros(new_size, new_size, channels, 'like', img);

    % Calcola la posizione di partenza per centrare l'immagine originale
    row_start = floor((new_size - height) / 2) + 1;
    col_start = floor((new_size - width) / 2) + 1;

    % Copia l'immagine originale nel centro della nuova immagine quadrata
    img_padded(row_start:row_start + height - 1, col_start:col_start + width - 1, :) = img;
end
