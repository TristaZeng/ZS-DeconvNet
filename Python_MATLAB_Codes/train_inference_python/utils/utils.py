import numpy as np


def prctile_norm(x, min_prc=0, max_prc=100):
    y = (x-np.percentile(x, min_prc))/(np.percentile(x, max_prc)-np.percentile(x, min_prc)+1e-7)
    y[y > 1] = 1
    y[y < 0] = 0
    return y

def read_mrc(filename, filetype='image'):
    rec_header_dtd = \
    [
        ("nx", "i4"),              # Number of columns
        ("ny", "i4"),              # Number of rows
        ("nz", "i4"),              # Number of sections

        ("mode", "i4"),            # Types of pixels in the image. Values used by IMOD:
                                   #  0 = unsigned or signed bytes depending on flag in imodFlags
                                   #  1 = signed short integers (16 bits)
                                   #  2 = float (32 bits)
                                   #  3 = short * 2, (used for complex data)
                                   #  4 = float * 2, (used for complex data)
                                   #  6 = unsigned 16-bit integers (non-standard)
                                   # 16 = unsigned char * 3 (for rgb data, non-standard)

        ("nxstart", "i4"),         # Starting point of sub-image (not used in IMOD)
        ("nystart", "i4"),
        ("nzstart", "i4"),

        ("mx", "i4"),              # Grid size in X, Y and Z
        ("my", "i4"),
        ("mz", "i4"),

        ("xlen", "f4"),            # Cell size; pixel spacing = xlen/mx, ylen/my, zlen/mz
        ("ylen", "f4"),
        ("zlen", "f4"),

        ("alpha", "f4"),           # Cell angles - ignored by IMOD
        ("beta", "f4"),
        ("gamma", "f4"),

        # These need to be set to 1, 2, and 3 for pixel spacing to be interpreted correctly
        ("mapc", "i4"),            # map column  1=x,2=y,3=z.
        ("mapr", "i4"),            # map row     1=x,2=y,3=z.
        ("maps", "i4"),            # map section 1=x,2=y,3=z.

        # These need to be set for proper scaling of data
        ("amin", "f4"),            # Minimum pixel value
        ("amax", "f4"),            # Maximum pixel value
        ("amean", "f4"),           # Mean pixel value

        ("ispg", "i4"),            # space group number (ignored by IMOD)
        ("next", "i4"),            # number of bytes in extended header (called nsymbt in MRC standard)
        ("creatid", "i2"),         # used to be an ID number, is 0 as of IMOD 4.2.23
        ("extra_data", "V30"),     # (not used, first two bytes should be 0)

        # These two values specify the structure of data in the extended header; their meaning depend on whether the
        # extended header has the Agard format, a series of 4-byte integers then real numbers, or has data
        # produced by SerialEM, a series of short integers. SerialEM stores a float as two shorts, s1 and s2, by:
        # value = (sign of s1)*(|s1|*256 + (|s2| modulo 256)) * 2**((sign of s2) * (|s2|/256))
        ("nint", "i2"),            # Number of integers per section (Agard format) or number of bytes per section (SerialEM format)
        ("nreal", "i2"),           # Number of reals per section (Agard format) or bit
                                   # Number of reals per section (Agard format) or bit
                                   # flags for which types of short data (SerialEM format):
                                   # 1 = tilt angle * 100  (2 bytes)
                                   # 2 = piece coordinates for montage  (6 bytes)
                                   # 4 = Stage position * 25    (4 bytes)
                                   # 8 = Magnification / 100 (2 bytes)
                                   # 16 = Intensity * 25000  (2 bytes)
                                   # 32 = Exposure dose in e-/A2, a float in 4 bytes
                                   # 128, 512: Reserved for 4-byte items
                                   # 64, 256, 1024: Reserved for 2-byte items
                                   # If the number of bytes implied by these flags does
                                   # not add up to the value in nint, then nint and nreal
                                   # are interpreted as ints and reals per section

        ("extra_data2", "V20"),    # extra data (not used)
        ("imodStamp", "i4"),       # 1146047817 indicates that file was created by IMOD
        ("imodFlags", "i4"),       # Bit flags: 1 = bytes are stored as signed

        # Explanation of type of data
        ("idtype", "i2"),          # ( 0 = mono, 1 = tilt, 2 = tilts, 3 = lina, 4 = lins)
        ("lens", "i2"),
        ("nd1", "i2"),             # for idtype = 1, nd1 = axis (1, 2, or 3)
        ("nd2", "i2"),
        ("vd1", "i2"),             # vd1 = 100. * tilt increment
        ("vd2", "i2"),             # vd2 = 100. * starting angle

        # Current angles are used to rotate a model to match a new rotated image.  The three values in each set are
        # rotations about X, Y, and Z axes, applied in the order Z, Y, X.
        ("triangles", "f4", 6),    # 0,1,2 = original:  3,4,5 = current

        ("xorg", "f4"),            # Origin of image
        ("yorg", "f4"),
        ("zorg", "f4"),

        ("cmap", "S4"),            # Contains "MAP "
        ("stamp", "u1", 4),        # First two bytes have 17 and 17 for big-endian or 68 and 65 for little-endian

        ("rms", "f4"),             # RMS deviation of densities from mean density

        ("nlabl", "i4"),           # Number of labels with useful data
        ("labels", "S80", 10)      # 10 labels of 80 charactors
    ]

    fd = open(filename, 'rb')
    header = np.fromfile(fd, dtype=rec_header_dtd, count=1)
    
    nx, ny, nz = header['nx'][0], header['ny'][0], header['nz'][0]

    if header[0][3] == 1:
        data_type = 'int16'
    elif header[0][3] == 2:
        data_type = 'float32'
    elif header[0][3] == 4:
        data_type = 'single'
        nx = nx * 2
    elif header[0][3] == 6:
        data_type = 'uint16'
    
    imgrawdata = np.fromfile(fd, data_type)
    fd.close()

    if filetype == 'image':
        data = np.ndarray(shape=(nx, ny, nz),dtype=data_type)
        for iz in range(nz):
            data_2d = imgrawdata[nx*ny*iz:nx*ny*(iz+1)]
            data[:, :, iz] = data_2d.reshape(nx, ny, order='F')
    else:
        data = imgrawdata

    return header, data