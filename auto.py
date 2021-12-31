import os
import argparse
from cryptography.fernet import Fernet

ap = argparse.ArgumentParser()
ap.add_argument('-t', '--task', type = str, help = 'encrypt / decrypt / publish')
ap.add_argument('-k', '--key', type = str, help = 'key')
ap.add_argument('-p', '--publish', type = str, help = 'test / original')
args = vars(ap.parse_args())

filesTo = ['./setup.py']

def create_package(args):
    os.system('rm -rf dist')
    os.system('rm rf styletx.egg-info')
    os.system('rm -rf build')

    run = ''
    if(args['publish'] == None):
        print('Not enough arguments. -h for help')
        exit()

    elif(args['publish'] == 'test'):
        run = 'twine upload --repository-url https://test.pypi.org/legacy/ dist/*'

    elif(args['publish'] == 'main'):
        run = 'twine upload dist/*'

    else:
        print('Invalid publish argument | test or main')
        exit()
    os.system('python3 setup.py sdist bdist_wheel')
    os.system(run)


def encrypt(args):

    if(args['key'] == None):
        print('Key not provided')
        exit()

    for fi in filesTo:
        with open(fi, 'r+') as f:
            s = f.read().encode('utf-8')
            f.seek(0)

            key = args['key'].encode('utf-8')
            key = Fernet(key)
            encrypted_text = key.encrypt(s)

            f.write(encrypted_text.decode('utf-8'))


def decrypt(args):

    if(args['key'] == None):
        print('Key not provided')
        exit()

    for fi in filesTo:
        try:
            with open(fi, 'r+') as f:
                s = f.read().encode('utf-8')
                f.seek(0)

                key = args['key'].encode('utf-8')
                key = Fernet(key)
                decrypted_text = key.decrypt(s)

                with open(fi, 'w') as ff:
                    ff.write(decrypted_text.decode('utf-8'))
        except:
            print("Something went wrong while decrypting")

if __name__ == '__main__':

    if(args['task'] == 'publish'):
        create_package(args)

    elif(args['task'] == 'encrypt'):
        encrypt(args)

    elif(args['task'] == 'decrypt'):
        decrypt(args)

    else:
        print('Invalid task argument| encrypt or decrypt or publish')
        exit()
