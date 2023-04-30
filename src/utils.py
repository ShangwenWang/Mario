def findAllIndex(src, key):
    res = []
    for i in range(src.__len__()):
        if src[i] == key:
            res.append(i)
    return res


def getMethodsInterval(src, key=','):
    res = []
    pushedFlag = True
    start = 0
    for i in range(src.__len__()):
        if src[i] != key and pushedFlag is True:
            start = i
            pushedFlag = False
        if src[i] == key and pushedFlag is False:
            res.append((start, i))
            pushedFlag = True
    if start < src.__len__() and pushedFlag is False:
        res.append((start, src.__len__()))
    return res

if __name__ == '__main__':
    assert getMethodsInterval([',', ','], ',') == []
    assert getMethodsInterval(['2', '444'], ',') == [(0, 2)]
    assert getMethodsInterval(['2', '444', ',', ','], ',') == [(0, 2)]
    assert getMethodsInterval(['1', '2', ',', '2', '444', ','], ',' ) == [(0, 2), (3, 5)]