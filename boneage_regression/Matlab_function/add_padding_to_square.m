function img_padded = add_padding_to_square(img)
    % Obtain image size
    [height, width, channels] = size(img);

    % Compute the box dimension (it will have side lenght equal to the
    % maximum of the image dimension)
    new_size = max(height, width);

    % Create an all black (every pixel is 0) squared image
    img_padded = zeros(new_size, new_size, channels, 'like', img);

    % Compute the position to center the image
    row_start = floor((new_size - height) / 2) + 1;
    col_start = floor((new_size - width) / 2) + 1;

    % Copy the original image in the center of the squared one
    img_padded( ...
        row_start:row_start + height - 1, ...
        col_start:col_start + width - 1, ...
        : ...
        ) = img;
end
