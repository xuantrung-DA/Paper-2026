import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=False)
    args = parser.parse_args()
    print('Training stub - config=', args.config)

if __name__ == '__main__':
    main()
