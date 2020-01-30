from subprocess import Popen

if __name__ == '__main__':
    '''
    for i in range(10054,  11851): # carcinoma valid
        src = 'carcinoma_train/{}.tif'.format(str(i).zfill(8))
        dst = 'carcinoma_valid/{}.tif'.format(str(i).zfill(8))
        p = Popen(['mv', src, dst])
    

    for i in range(11851,  14160): # carcinoma test
        src = 'carcinoma_train/{}.tif'.format(str(i).zfill(8))
        dst = 'carcinoma_test/{}.tif'.format(str(i).zfill(8))
        p = Popen(['mv', src, dst])
    '''

    for i in range(6495,  7912): # lymphoma valid
        src = 'lymphoma_train/{}.tif'.format(str(i).zfill(8))
        dst = 'lymphoma_valid/{}.tif'.format(str(i).zfill(8))
        p = Popen(['mv', src, dst])
    

    for i in range(7912,  9360): # lymphoma test
        src = 'lymphoma_train/{}.tif'.format(str(i).zfill(8))
        dst = 'lymphoma_test/{}.tif'.format(str(i).zfill(8))
        p = Popen(['mv', src, dst])