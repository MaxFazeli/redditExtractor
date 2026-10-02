# extract subreddit (submissions+comments) from reddit data dump
# (https://files.pushshift.io/reddit/)
# last update: 17.08.2023

import os
import datetime
import bz2
import lzma
import multiprocessing
import shutil
import subprocess
import sys



def create_result_folders(foldername):
    if not os.path.exists("comments/"+foldername):
        os.makedirs("comments/"+foldername)
    if not os.path.exists("submissions/" + foldername):
        os.makedirs("submissions/" + foldername)


def debug_msg(msg):
    print ("  \x1b[33m-PPRZ-\x1b[00m [\x1b[36m{}\x1b[00m] " \
          "\x1b[35m{}\x1b[00m".format(datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"), msg))


def get_files_from_folder(folder, filetype):
    import glob
    return glob.glob(folder+filetype)


def load_files(path):
    log_files = []
    for path, subdirs, files in os.walk(path):
        for name in files:
            log_files.append(os.path.join(path, name))
    return log_files


def load_source_files(fpath):
    log_files = load_files(fpath)
    return sorted(log_files)#, reverse=True)


def read_file_mp(params):
    (source_path, subreddit_filter, subreddit, id, total, entity_type) = params
    submissions = []
    year = source_path.split("_")[-1][:4]
    month = source_path.split("-")[-1].split(".")[0]

    destination_path_without_ending = entity_type+"/"+subreddit+"/"+subreddit+"_{}_{}".format(year, month)
    # check if file was already processed in previous runs:
    if os.path.isfile(destination_path_without_ending + ".txt"):
        debug_msg("[{}/{}] Already processed in previous runs: {}".format(id, total, source_path))
        return

    debug_msg("[{}/{}] Processing: {}".format(id, total, source_path))
    if source_path.endswith(".bz2"):
        input_file = bz2.open(source_path, "rt")
        # month = source_path.split("-")[-1].replace(".bz2", "")
    elif source_path.endswith(".xz"):
        input_file = lzma.open(source_path, "rt")
        # month = source_path.split("-")[-1].replace(".xz", "")
    elif source_path.endswith(".zst"):

        
        source_filename = source_path.split("/")[-1]
        source_filename_without_ending = source_filename.replace(".zst", "")
        # remove tmp source files from previous runs
        if os.path.isfile(source_filename): os.remove(source_filename)
        if os.path.isfile(source_filename_without_ending): os.remove(source_filename_without_ending)
        # copy .zst file:
        shutil.copyfile(source_path, source_filename)
        
        # unzip .zst file:
        with open(os.devnull, 'wb') as devnull:
            subprocess.check_call(['zstd', '-d', source_filename, '--long=31'], stdout=devnull, stderr=subprocess.STDOUT)
        # remove .zst file:
        if os.path.isfile(source_filename):
            os.remove(source_filename)
        input_file = open(source_filename_without_ending, "rt")
        # month = source_path.split("-")[-1].replace(".zst", "")
    f = open(destination_path_without_ending + ".TMP", "wt") # call the output file .tmp first, rename it later for ignoring already processed files in further runs

    # to avoid memory problems: read line per line instead of whole file:
    # for l in input_file.readlines():
    while True:
        l = input_file.readline()
        if l == "":
            # avoid stopping at empty lines inbetween the file
            for i in range(1000):
                l = input_file.readline()
                if l != "":
                    break
            if l == "":
                break
        if subreddit_filter in l:
            submissions.append(l.strip())
            if len(submissions) > 1000:
                f.write(("\n").join(submissions)+"\n")
                submissions = []
    
    # remove unzipped file (only for .zst):
    if source_path.endswith(".zst"):
        if os.path.isfile(source_filename_without_ending):
            os.remove(source_filename_without_ending)

    f.write(("\n").join(submissions)+"\n")
    f.close()
    input_file.close()
    del f
    del input_file
    os.rename(destination_path_without_ending + ".TMP", destination_path_without_ending + ".txt")
    debug_msg("[{}/{}]   ++ done [{}]".format(id, total, source_path))
    return
    #return submissions


def concat_files(subreddit, entity_type, delete_source=False):
    outfile = entity_type+"/"+subreddit+"/"+subreddit+"_"+entity_type+".txt"
    if os.path.isfile(outfile):
        debug_msg("Final output file {} already exists.".format(outfile))
    else:
        files = load_source_files(entity_type+"/"+subreddit+"/")
        with open(outfile, "wt") as output:
            for f in files:
                if f == outfile:
                    continue
                with open(f, "rt") as fd:
                    shutil.copyfileobj(fd, output, 1024*1024*10)
                    output.write("\n")
        debug_msg("Final output file {} written.".format(outfile))
    if delete_source:
        for file in files:
            if file == outfile:
                continue
            os.remove(file)
        debug_msg("txt files deleted.".format(outfile))
    return


def extract_entities_for_subreddit(subreddit, comments_files_path, entity_type="submissions", num_processes=24, del_src=False):
    lfiles = load_source_files(comments_files_path)
    subreddit_filter = "\"subreddit\":\"{}\"".format(subreddit)
    params = []
    debug_msg("Starting to parse files!")
    for fdx, lf in enumerate(lfiles):
        params.append([lf, subreddit_filter, subreddit, fdx, len(lfiles), entity_type])
    p = multiprocessing.Pool(num_processes)
    p.map(read_file_mp, (params))
    p.close()
    p.join()
    debug_msg("Done parsing files!")
    debug_msg("Concatenating files!")
    concat_files(subreddit, entity_type, delete_source=del_src)
    debug_msg("Done")


def main():
    nr_processes = multiprocessing.cpu_count()
    nr_processes = 12
    sub_path = "/media/nas_datasets/datasets/Reddit/submissions/"
    com_path = "/media/nas_datasets/datasets/Reddit/comments/"

    subreddits = ["ifyoulikeblank"]
    for subreddit in subreddits:
        create_result_folders(subreddit)
        extract_entities_for_subreddit(subreddit, sub_path, "submissions", nr_processes, False)
        extract_entities_for_subreddit(subreddit, com_path, "comments", nr_processes, False)


if __name__ == "__main__":
    main()
