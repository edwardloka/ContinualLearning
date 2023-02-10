from PIL import Image, ImageFilter, ImageTk, ImageGrab
import numpy as np

def convertImage(filename):

    img = Image.open(filename).convert('L')
    width = float(img.size[0])
    height = float(img.size[1])
    newImage = Image.new('L', (28,28), (255))

    if width > height :

        nheight = int(round((20.0 / width * height), 0))
        nheight = 1 if nheight == 0 else nheight

        img = img.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))
        newImage.paste(img, (4, wtop))

    else:
        nwidth = int(round((20.0 / height * width), 0))
        if nwidth == 0:
            nwidth = 1
        nwidth = 1 if nwidth == 0 else nwidth

        img = img.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))
        newImage.paste(img, (wleft, 4))

    tv = list(newImage.getdata())

    tva = [(255 - x) for x in tv]

    return tva

def convert(filename):
    # x = [convertImage(filename)]
    # newArr = [[0 for d in range(28)] for y in range(28)]
    # k = 0
    # for i in range(28):
    #     for j in range(28):
    #         newArr[i][j] = x[0][k]
    #         k = k + 1
    #
    # plt.imshow(newArr, interpolation='nearest')
    # plt.savefig('MNIST_IMAGE.png')
    # plt.show()

    result = np.array(convertImage(filename))
    return result

#print(convert('images/out.png'))