import arxiv #requirement
from time import sleep
import os


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "query",
        help="keywords to query arxiv database"
    )

    parser.add_argument(
        "--max-results",
        help="max result set size",
        default=1000
    )

    parser.add_argument(
        "--batch-size",
        help="batch of paper downloaded without sleep",
        default=5,
    )

    parser.add_argument(
        "--dir",
        help="download destination folder",
        default="./downloaded/",
    )

    parser.add_argument(
        "--tts",
        help="time to sleep after a batch has been downloaded",
        default=30
    )

    parser.add_argument(
        "--offset",
        help="query offset",
        default=0
    )

    args = parser.parse_args()

    if(not os.path.isdir(args.dir)): os.makedirs(args.dir,exist_ok=True)

    print(
        '''summary
        \tquery string: {}
        \tdestination folder: {}
        \tresuls: {}
        \ttime to sleep(s): {}\n
        '''.format(args.query,args.dir,args.max_results,args.tts)
    )
    
    max_results = int(args.max_results)
    offset = int(args.offset)
    batch_size = int(args.batch_size)
    tts = int(args.tts)

    querySet = arxiv.query(
    query=args.query,
    id_list=[],
    max_results=max_results,
    start = offset,
    sort_by="relevance",
    sort_order="descending",
    prune=True,
    iterative=False,
    max_chunk_results=1000
    )

    res_num = len(querySet)

    failed = 0
    successes = 0

    printProgressBar(0, batch_size, prefix = 'Progress batch [{},{}]:'.format(0,batch_size), suffix = 'Complete', length = 50)
    for j in range(0,res_num,batch_size):
        # print("started donwloading {}:{}...".format(j,j+batch_size),end="")
        for l,paper in enumerate(querySet[j:j+batch_size]):
            try:
                arxiv.download(paper, dirpath=args.dir)
                successes+=1
            except:
                failed += 1
                pass
            printProgressBar((j+l+1)%batch_size+1, batch_size, prefix = 'Progress batch [{}/{}]:'.format((j+batch_size)/batch_size,res_num/batch_size), suffix = '[{} failed][{} downloaded]'.format(failed,successes), length = 50)
        sleep(tts)