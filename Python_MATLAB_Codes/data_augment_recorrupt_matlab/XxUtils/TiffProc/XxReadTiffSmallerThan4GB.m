function data = XxReadTiffSmallerThan4GB(TiffPath,start_frame,end_frame,interval)

pic_info = imfinfo(TiffPath);
[frame_CCD,~] = size(pic_info);
if nargin < 4, interval = 1; end
if nargin < 3, end_frame = frame_CCD; end
if nargin < 2, start_frame = 1; end
row_CCD = pic_info.Height;
colum_CCD = pic_info.Width;

temp=pic_info.PhotometricInterpretation;
if strfind(temp,'RGB')==1
    data=zeros(row_CCD,colum_CCD,3,round((end_frame-start_frame+1)/interval));%得到图像矩阵的各个参数，并提前赋值好零矩阵
    for frame_temp=start_frame:interval:end_frame
        data(:,:,:,round((frame_temp-start_frame)/interval)+1)=imread(TiffPath,'Index',frame_temp);
    end
else
    data = zeros(row_CCD,colum_CCD,round((end_frame-start_frame+1)/interval));
    for frame_temp = start_frame:interval:end_frame
        data(:,:,round((frame_temp-start_frame)/interval)+1) = imread(TiffPath,'Index',frame_temp);
    end
end

end