function output = XxGetIntesPrctile(data,prc)
    intens = prctile(data(:),prc);
    output = mean(data(data > intens));
end