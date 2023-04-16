function XxWriteTiff(data, FileName)

imwrite(data(:,:,1), FileName, 'Compression','none');

if size(data,3)>1
    for i=2:size(data,3)
        imwrite(data(:,:,i), FileName, 'WriteMode', 'append', 'Compression','none');
    end
end
