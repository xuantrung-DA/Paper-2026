import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=False)
    args = parser.parse_args()
    print('Eval stub - ckpt=', args.ckpt)

if __name__ == '__main__':
    main()
