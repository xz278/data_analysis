# coding=utf-8
"""
Convert file encdoing.
"""
import codecs
import argparse

def convert_encoding(source_f, source_encoding,
              target_f, target_encoding,
              block_size=1048576):
    """
    Convert encoding of a file.

    Parameters:
    -----------
    source_f: str
        Source file.

    source_encoding: str
        Encoding system of source file.
        'gb18030' is the suggested encoding system
        if the source is from MS Windows(CHN) ANSI.

    target_f: str
        Name of the file to store the converted data.
        Default is None, in which case the encoding is
        appended to the source file name.

    target_encoding: str
        Target encoding system.

    block_size: float
        Desired block size in bytes to read a file.

    Returns:
    --------

    """
    if target_f is None:
        file_extension = source_f.split('.')[-1]
        source_f_len = len(source_f) - len(file_extension)
        source_file_name = source_f[:(source_f_len - 1)]
        target_f = '{}-{}.{}'.format(source_file_name,
                            target_encoding,
                            file_extension)
    with codecs.open(source_f, "r", source_encoding) as sourceFile:
        with codecs.open(target_f, "w", target_encoding) as targetFile:
            while True:
                contents = sourceFile.read(block_size)
                if not contents:
                    break
                targetFile.write(contents)


def main():
    """
    Handle command line opitons.
    """
    parser = argparse.ArgumentParser()

    # commond
    parser.add_argument('-sf', '--source_file', required=True,
                        help='Source file')

    parser.add_argument('-se', '--source_encoding', required=True,
                        help='Source encoding')

    parser.add_argument('-tf', '--target_file', default=None,
                        help='Target file')

    parser.add_argument('-te', '--target_encoding', required=True,
                        help='Target encoding')

    parser.add_argument('-bs', '--block_size', default=1048576,
                        help='Block size')

    args = parser.parse_args()

    # convert encoding
    convert_encoding(args.source_file, args.source_encoding,
                args.target_file, args.target_encoding,
                args.block_size)

if __name__ == '__main__':
    main()