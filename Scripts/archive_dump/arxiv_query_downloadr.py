import arxiv #requirement
from time import sleep

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



max_results = 1000
offset = 110
batch_size = 250

querySet = arxiv.query(
  query="machine learning",
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

print("querySet: \t\nmax_res: {}\n\tlen: {}\n\toffset: {}\n".format(max_results,res_num,type(querySet),type(querySet[0])))

failed = 0
successes = 0

printProgressBar(0, batch_size, prefix = 'Progress batch [{},{}]:'.format(0,batch_size), suffix = 'Complete', length = 50)
for j in range(0,res_num,batch_size):
    # print("started donwloading {}:{}...".format(j,j+batch_size),end="")
    for l,paper in enumerate(querySet[j:j+batch_size]):
        try:
            arxiv.download(paper, dirpath="./downloaded/")
            successes+=1
        except:
            failed += 1
            pass
        printProgressBar((j+l+1)%batch_size+1, batch_size, prefix = 'Progress batch [{}/{}]:'.format((j+batch_size)/batch_size,res_num/batch_size), suffix = '[{} failed][{} downloaded]'.format(failed,successes), length = 50)
    sleep(60)

