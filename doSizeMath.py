

def findConv2dSize(height,kernalSize,stride, padding,dialation=1):
    x = height+2*padding - dialation * (kernalSize-1) -1
    x = x // stride
    print(x +1)

def findDeConv2dSize(height, kernalSize, stride, padding, outputPadding = 0, dialation=1):
    x = (height -1) * stride -2 * padding + dialation * (kernalSize -1) + outputPadding + 1
    print(x)


def main():
    # findConv2dSize(28,4,2,0)
    # findConv2dSize(13,3,2,1)
    # findConv2dSize(7,3,2,1)
    # findConv2dSize(4,2,2,1)

    findDeConv2dSize(1,4,1,0)
    findDeConv2dSize(4,3,2,1)
    findDeConv2dSize(7,3,2,1)
    findDeConv2dSize(13,4,2,0)
    # findDeConv2dSize(824,4,2,1)








if __name__ == "__main__":
    main()