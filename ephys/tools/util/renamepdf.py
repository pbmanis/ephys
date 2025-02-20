from pathlib import Path

def main():
    fns = Path('.').glob('nf107_maps*.pdf')
    for f in fns:
        print('file: ', f)
        fs = str(f)
        fnew = fs.replace('nf107', 'NF107Ai32Het')
        print('fnew: ', fnew)
        f.rename(fnew)
    
    
if __name__ == '__main__':
    main()
