import os
import numpy as np


add_dll_dir = getattr(os, "add_dll_directory", None)
vipsbin = r"C:\AI\vips-dev-8.14\bin"
if callable(add_dll_dir):
    add_dll_dir(vipsbin)    
else:
    os.environ["PATH"] = os.pathsep.join((vipsbin, os.environ["PATH"]))
   
format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

# map np dtypes to vips
dtype_to_format = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
}


import pyvips


#list the tiles to be stitched 
imgs = os.listdir(pathToMyTiles)


#slide dimensions are extracted from the filenames of the tiles
slideWidth,slideHeight = getWidthAndHeight(imgs)


result = pyvips.Image.black(slideWidth,slideHeight,bands=3)
print("result format " + result.format)


for img in imgs: 

    print("-----------------")

    tile = pyvips.Image.new_from_file(img, access='sequential')
    mem_img = tile.write_to_memory()

    print("tile format " + str(tile.format))
    print("tile dtype " + str(format_to_dtype[tile.format]))

    #convert to numpy
    np_3d = np.ndarray(buffer=mem_img,dtype=format_to_dtype[tile.format],shape=[tile.height, tile.width, tile.bands])
    height, width, bands = np_3d.shape
    linear = np_3d.reshape(width * height * bands)


    print("Numpy array DataType " + str(np_3d.dtype))
    print("numpy array format format "+ str(dtype_to_format[str(np_3d.dtype)]))

    #make some changes
    np_3d[:,:,0] = 220

    #reconvert to vips image
    vi = pyvips.Image.new_from_memory(linear.data, width, height, bands,
                                          dtype_to_format[str(np_3d.dtype)])

    print("format after going back to vips  " + str(vi.format))
    

    

    #tile position is extracted from the filename 
    absX,absY = extractTileCoordinates(img)

    result = result.insert(vi,absX,absY)

    result.tiffsave(safePath, compression=pyvips.enums.ForeignTiffCompression.DEFLATE,
                 tile=True, tile_width=512, tile_height=512, #rgbjpeg=True,
                 pyramid=True,  bigtiff=True)

    print("-----------------")








