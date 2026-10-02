import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
from collections import defaultdict
import csv
import datetime
from difflib import SequenceMatcher
import io
import json
from logger import debug_msg, DebugLevel
from math import log
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os
import pandas as pd
import pickle
from random import randint
import re
import seaborn as sns
from statsmodels.stats.anova import AnovaRM
import sys
from tools import getDurationString

DIRECTORY = os.path.dirname(os.path.realpath(__file__))
DATA_DIRECTORY = os.path.join(DIRECTORY, "data")
SUBMISSIONID_PARAMETERS_FILENAME = "submissionid_parameters.obj"
SUBMISSIONID_COMMENTSMATCHED_FILENAME = "submissionid_commentsmatched.obj"
SUBMISSIONID_TRUEMOVIETITLES_FILENAME = "submissionid_truemovietitles.obj"
SUBMISSIONID_TRUEMOVIETITLES_MULTILANGUAGE_FILENAME = "submissionid_truemovietitles__MULTI_LANGUAGE_TITLES.obj"
MOVIE_TITLES_FILENAMES = ["movie_titles_international__large.csv", "movie_titles__large.csv", "movie_titles.csv"]
TTID2MOVIE_TITLE_SET_FILENAME = "all_imdb_ttid2movietitle_set.obj"
RESULTS_DIRECTORY = os.path.join(DIRECTORY, "results")
RESULTS_DIRECTORY_PREFIXES = ["results_ollama__", "results_gpt__"]
CYCLE_DIRECTORY_PREFIXES = ["ollama__", "gpt__"]
RESPONSE_FILES_SUFFIX = "_response.json"
REQUEST_FILES_SUFFIX = "_request.json"
EXTRACTION_RESULTS_FILENAME = "results_extraction.csv"
EXTRACTION_RESULTS_MEDIAN_CYCLES_FILENAME = "results_extraction__median_cycles.csv"
EVALUATION_RESULTS_FILENAME = "results_evaluation.csv"
EVALUATION_RESULTS_MEDIAN_CYCLES_FILENAME = "results_evaluation__median_cycles.csv"
EVALUATION_RESULTS_MULTI_LANGUAGE_FILENAME = "results_evaluation__MULTI_LANGUAGE.csv"
EVALUATION_RESULTS_MULTI_LANGUAGE_MEDIAN_CYCLES_FILENAME = "results_evaluation__MULTI_LANGUAGE__median_cycles.csv"
NR_RECOMMENDATIONS = 10
NR_BOOTSTRAP_SAMPLES = int(1e5)
CI_FORMAT_STRING = "$^{[%.4f, %.4f]}$"

# avoid type 3 fonts:
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.sans-serif'] = "Linux Biolinum O"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams["mathtext.fontset"] = "custom"


def loadPickle(path_to_file):
    try:
        with open(path_to_file, 'rb') as infile:
            data = pickle.load(infile)
    except:
        data = None
    return data


def savePickle(data, path_to_file):
    with open(path_to_file, 'wb') as outfile:
        pickle.dump(data, outfile, -1)
    return


def readSubmissionsFromFile(directory=".", filename="MovieSuggestions_submissions_pretty.txt", subreddit_id="t5_2sqpa"):
    submissions = list()
    with io.open(directory + "/data/" + filename, encoding="utf-8-sig") as infile:
        content = infile.read().split("\n" + subreddit_id)
        for line in content[1:]:
            line = (subreddit_id + line).split("\t")
            submission = dict()
            submission["subredditId"] = line[0].strip()
            submission["subreddit"] = line[1].strip()
            submission["created_at"] = line[2].strip()
            submission["author"] = line[3].strip()
            submission["num_comments"] = int(line[4].strip())
            submission["url"] = line[5].strip()
            submission["score"] = int(line[6].strip())
            submission["link_flair_css_class"] = line[7].strip()
            submission["title"] = line[8].strip()
            submission["selftext"] = line[9].strip()
            try:
                submission["link"] = line[10].strip()
            except:
                submission["link"] = submission["url"]
            # other_fields:
            link = submission["link"]
            begin = link.find("comments") + 9
            end = link[begin:].find("/")
            submission["id"] = link[begin:begin + end]
            submissions.append(submission)
    return submissions


def getSubmissionById(submissions, submission_id):
    for submission in submissions:
        if submission['id'] == submission_id:
            return submission
    return None


def isTitleSimilarityAboveThreshold(list, item, threshold):
    for l in list:
        ratio = SequenceMatcher(None, item, l).ratio()
        if ratio >= threshold:
            return True
    return False


def prediction_metrics(true, pred, threshold=None):
    tp = 0
    fp = 0
    fn = 0
    DCG = 0.0
    IDCG = 0.0
    index = 0
    for p in pred:
        index += 1
        if p in true or (threshold is not None and isTitleSimilarityAboveThreshold(true, p, threshold)):
            tp += 1
            DCG += 1.0 / log(index + 1, 2)
        else:
            fp += 1
    if threshold is None:  # this is unnecessary!
        for t in true:
            if t not in pred:
                fn += 1
    else:
        fn = len(true) - tp
    for i in range(min(len(true), len(pred))):
        i += 1
        IDCG += 1.0 / log(i + 1, 2)

    precision = 1.0 * tp / (tp + fp) if (tp + fp) else 0.0
    recall = 1.0 * tp / (tp + fn) if (tp + fn) else 0.0
    f1_score = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    nDCG = 1.0 * DCG / IDCG if IDCG else 0.0
    return precision, recall, f1_score, nDCG


def prediction_metrics_multi_language(true, pred, threshold=None):
    tp = 0
    fp = 0
    DCG = 0.0
    IDCG = 0.0
    index = 0
    for p in pred:
        index += 1
        found = False
        for true_ in true:
            if p in true_ or (threshold is not None and isTitleSimilarityAboveThreshold(true_, p, threshold)):
                tp += 1
                DCG += 1.0 / log(index + 1, 2)
                found = True
                break
        if not found:
            fp += 1
    fn = len(true) - tp
    for i in range(min(len(true), len(pred))):
        i += 1
        IDCG += 1.0 / log(i + 1, 2)

    precision = 1.0 * tp / (tp + fp) if (tp + fp) else 0.0
    recall = 1.0 * tp / (tp + fn) if (tp + fn) else 0.0
    f1_score = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    nDCG = 1.0 * DCG / IDCG if IDCG else 0.0
    return precision, recall, f1_score, nDCG


def readTrueMovieTitlesOfSubmission(submission_id,
                                    submissionid_parameters,
                                    submissionid_commentsmatched,
                                    ttid2movietitle):
    true_ttids = set()
    for comment_id, ttids in submissionid_commentsmatched[submission_id].items():
        true_ttids |= set(ttids)
    # remove input ttids:
    for ttid in list(true_ttids):
        if ttid in " ".join(submissionid_parameters[submission_id]):
            true_ttids.remove(ttid)

    true_movie_titles = list()
    for ttid in true_ttids:
        # convert "Title (2020/II)" to "Title (2020)" OR "Title (2020/II) (TV)" to "Title (2020)":
        movie_title_and_year = re.findall("(.+?)\s\((\d{4})(?:\/\S+?)?\)(?:\s\(\S+\))?$", ttid2movietitle[ttid])
        assert (len(movie_title_and_year) == 1)
        title_new = (movie_title_and_year[0][0] + " (" + movie_title_and_year[0][1] + ")").strip()
        true_movie_titles.append(title_new)
    return sorted(true_movie_titles)


def readTrueMovieTitlesMultiLanguageOfSubmission(submission_id,
                                                 submissionid_parameters,
                                                 submissionid_commentsmatched,
                                                 ttid2movietitle,
                                                 ttid2movietitle_set):
    true_ttids = set()
    for comment_id, ttids in submissionid_commentsmatched[submission_id].items():
        true_ttids |= set(ttids)
    # remove input ttids:
    for ttid in list(true_ttids):
        if ttid in " ".join(submissionid_parameters[submission_id]):
            true_ttids.remove(ttid)

    # multiple language titles:
    true_movie_title_sets = list()
    for ttid in true_ttids:
        true_movie_titles = set()
        if ttid not in ttid2movietitle_set:  # because it's e.g. an episode of a series
            # convert "Title (2020/II)" to "Title (2020)" OR "Title (2020/II) (TV)" to "Title (2020)":
            try:
                movie_title_and_year = re.findall("(.+?)\s\((\d{4})(?:\/\S+?)?\)(?:\s\(\S+\))?$", ttid2movietitle[ttid])
            except:
                debug_msg("ttid not in ttid2movietitle: " + str(ttid), DebugLevel.DEBUG)
                continue
            assert (len(movie_title_and_year) == 1)
            title_new = (movie_title_and_year[0][0] + " (" + movie_title_and_year[0][1] + ")").strip()
            true_movie_titles.add(title_new)
        else:
            for title in ttid2movietitle_set[ttid]:
                # convert "Title (2020/II)" to "Title (2020)" OR "Title (2020/II) (TV)" to "Title (2020)":
                movie_title_and_year = re.findall("(.+?)\s\((\d{4})(?:\/\S+?)?\)(?:\s\(\S+\))?$", title)
                assert (len(movie_title_and_year) == 1)
                title_new = (movie_title_and_year[0][0] + " (" + movie_title_and_year[0][1] + ")").strip()
                true_movie_titles.add(title_new)
        true_movie_title_sets.append(true_movie_titles)
    return true_movie_title_sets


def createMovieTitlesCSVs(data_directory=DATA_DIRECTORY):
    submissionid_parameters = loadPickle(data_directory + "/" + SUBMISSIONID_PARAMETERS_FILENAME)
    submissionid_commentsmatched = loadPickle(data_directory + "/" + SUBMISSIONID_COMMENTSMATCHED_FILENAME)
    all_imdb_movie_titles_df = pd.read_csv(os.path.join(DATA_DIRECTORY, "all_imdb_movie_titles_df.csv"))

    ttids = set()
    for submission_id, parameters in submissionid_parameters.items():
        for p in parameters:
            ttid = p.split('"')[1]
            ttids.add(ttid)

    for submission_id, comments in submissionid_commentsmatched.items():
        for ttid in comments["comments"]:
            ttids.add(ttid)

    ttid2movietitle = dict()
    ttid2movietitle_international = dict()

    for ttid in ttids:
        title_of_ttid_df = all_imdb_movie_titles_df.loc[(all_imdb_movie_titles_df.ttid == ttid)]
        row = title_of_ttid_df.loc[(title_of_ttid_df.titleType == "originalTitle")]
        if len(row) > 0:
            ttid2movietitle[ttid] = str(row.title.values[0]) + " (" + str(row.year.values[0]) + ")"
        row = title_of_ttid_df.loc[(title_of_ttid_df.titleType == "primaryTitle")]
        if len(row) > 0:
            ttid2movietitle_international[ttid] = str(row.title.values[0]) + " (" + str(row.year.values[0]) + ")"
    with open(os.path.join(data_directory, "movie_titles.csv"), "w") as f:
        f.write("movie_id;movie_title\n")
        for ttid in sorted(ttid2movietitle):
            f.write(ttid + ";" + ttid2movietitle[ttid] + "\n")
    with open(os.path.join(data_directory, "movie_titles_international__large.csv"), "w") as f:
        f.write("movie_id;movie_title\n")
        for ttid in sorted(ttid2movietitle_international):
            f.write(ttid + ";" + ttid2movietitle_international[ttid] + "\n")
    with open(os.path.join(data_directory, "movie_titles__large.csv"), "w") as f:
        # this is an unused file:
        f.write("")

    fitting_submission_ids = set()
    all_ttids = set(ttid2movietitle) | set(ttid2movietitle_international)
    for submission_id, comments in submissionid_commentsmatched.items():
        # remove wrong ttids:
        for ttid in list(comments["comments"].copy()):
            if ttid not in all_ttids:
                submissionid_commentsmatched[submission_id]["comments"].remove(ttid)
        assert len(submissionid_commentsmatched[submission_id]["comments"]) == len(
            set(submissionid_commentsmatched[submission_id]["comments"]) & all_ttids)
        if len(set(submissionid_commentsmatched[submission_id]["comments"]) & all_ttids) < NR_RECOMMENDATIONS:
            continue
        fitting_submission_ids.add(submission_id)
    submission_ids = fitting_submission_ids
    savePickle(submission_ids,
               os.path.join(data_directory, "only_fitting/submission_ids__only_fitting_%d.obj" % len(submission_ids)))

    submission_ids_to_delete = (set(submissionid_commentsmatched.keys()) | set(submissionid_parameters.keys())) - set(
        submission_ids)
    for submission_id in submission_ids_to_delete:
        try:
            del submissionid_parameters[submission_id]
        except:
            pass
        try:
            del submissionid_commentsmatched[submission_id]
        except:
            pass
    assert (set(submissionid_commentsmatched.keys()) | set(submissionid_parameters.keys()) | set(submission_ids)) == (
            set(submissionid_commentsmatched.keys()) & set(submissionid_parameters.keys()) & set(submission_ids))
    savePickle(submissionid_parameters, os.path.join(data_directory, "only_fitting/submissionid_parameters.obj"))
    savePickle(submissionid_commentsmatched,
               os.path.join(data_directory, "only_fitting/submissionid_commentsmatched.obj"))
    return


def createAndSaveSubmission2trueMovieTitles(data_directory=DATA_DIRECTORY, submissionid_parameters=None,
                                            submissionid_commentsmatched=None):
    if submissionid_parameters is None:
        submissionid_parameters = loadPickle(data_directory + "/" + SUBMISSIONID_PARAMETERS_FILENAME)
    if submissionid_commentsmatched is None:
        submissionid_commentsmatched = loadPickle(data_directory + "/" + SUBMISSIONID_COMMENTSMATCHED_FILENAME)

    ttid2movietitle = dict()
    for movie_titles_filename in MOVIE_TITLES_FILENAMES:
        with open(data_directory + "/" + movie_titles_filename, "r") as file:
            csvreader = csv.reader(file, delimiter=";")
            for row in csvreader:
                if row[0] not in ttid2movietitle:
                    ttid2movietitle[row[0]] = row[1]

    submissionid2trueMovieTitles = dict()
    for submission_id in sorted(set(submissionid_commentsmatched.keys()) & set(submissionid_parameters.keys())):
        true_movie_titles_list = readTrueMovieTitlesOfSubmission(
            submission_id,
            submissionid_parameters,
            submissionid_commentsmatched,
            ttid2movietitle)
        submissionid2trueMovieTitles[submission_id] = true_movie_titles_list
    savePickle(submissionid2trueMovieTitles, data_directory + "/" + SUBMISSIONID_TRUEMOVIETITLES_FILENAME)

    # for multiple language titles:
    ttid2movietitle_set = loadPickle(os.path.join(DATA_DIRECTORY, TTID2MOVIE_TITLE_SET_FILENAME))

    submissionid2trueMovieTitles = dict()
    for submission_id in sorted(set(submissionid_commentsmatched.keys()) & set(submissionid_parameters.keys())):
        true_movie_titles_list = readTrueMovieTitlesMultiLanguageOfSubmission(
            submission_id,
            submissionid_parameters,
            submissionid_commentsmatched,
            ttid2movietitle,
            ttid2movietitle_set)
        submissionid2trueMovieTitles[submission_id] = true_movie_titles_list
    savePickle(submissionid2trueMovieTitles, data_directory + "/" + SUBMISSIONID_TRUEMOVIETITLES_MULTILANGUAGE_FILENAME)
    return


def getAllDirectories(directory=DIRECTORY, prefix=RESULTS_DIRECTORY_PREFIXES[0]):
    return sorted(
        [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d)) and d.startswith(prefix)])


def readResponseFiles(directory, suffix=RESPONSE_FILES_SUFFIX):
    filename2content = dict()
    for filename in sorted(
        [d for d in os.listdir(directory) if os.path.isfile(os.path.join(directory, d)) and d.endswith(suffix)]):
        with open(os.path.join(directory, filename)) as f:
            filename2content[filename] = json.load(f)
    return filename2content


def prepareContent(filename2content):
    filename2jsonResponses = dict()
    for filename, content in filename2content.items():
        try:
            response = ""
            if "message" in content:
                response = content["message"]["content"]
            elif "body" in content:
                assert (len(content["body"]["choices"]) == 1)
                response = content["body"]["choices"][0]["message"]["content"]
            else:
                debug_msg("ERROR: response broken? response: %s" % str(content), DebugLevel.WARNING)
            j = json.loads(response)
            filename2jsonResponses[filename] = j
        except Exception as e:
            filename2jsonResponses[filename] = False
    return filename2jsonResponses


def correctMovieTitle(movie_title):
    movie_title = movie_title.strip()
    movie_title_and_year = re.findall("(.+?)\s\((\d{4})\)", movie_title)
    if len(
        movie_title_and_year) == 1:  # e.g. "Sunshine (2007) bla (2007) 3 (2008) asdf" this will return 3 groups --> wrong
        if (movie_title_and_year[0][0] + " (" + movie_title_and_year[0][1] + ")").strip() == movie_title.strip():
            return (movie_title_and_year[0][0] + " (" + movie_title_and_year[0][1] + ")").strip()
    return None


def processMovieStr(movie_str):
    corrected_movie_title = correctMovieTitle(movie_str)
    if corrected_movie_title != None:
        return [corrected_movie_title]
    movie_titles_list = list()
    return movie_titles_list


def processMovieList(movie_list):
    movie_titles_list = list()
    for element in movie_list:
        movie_titles_list += processMovieItem(element)
    return movie_titles_list


def processMovieItem(movie_item):
    if type(movie_item) is str:
        return processMovieStr(movie_item)
    elif type(movie_item) is list:
        return processMovieList(movie_item)
    elif type(movie_item) is dict:
        return processMovieDict(movie_item)
    return list()


# We set that only strings (no matter whether in a list or in a dict...) with "movie title (year)" are valid
def processMovieDict(movie_dict):
    movie_titles_list = list()
    for k, v in movie_dict.items():
        movie_titles_list += processMovieItem(k)
        movie_titles_list += processMovieItem(v)
    return movie_titles_list


def evaluateCycle(model, shot, cycle, filename2jsonResponses, df):
    invalid_json_counter = 0
    for filename, json_response in filename2jsonResponses.items():
        movie_titles_list = list()
        if json_response == False:
            invalid_json_counter += 1
            valid_json = False
        else:
            assert (type(json_response) is dict or type(json_response) is list)
            valid_json = True
            movie_titles_list = processMovieItem(json_response)

        # remove duplicates case-insensitive (and keep ordering!):
        movie_titles_lower_set_without_duplicates = set()
        movie_titles_list_without_duplicates = list()
        for movie_title in movie_titles_list:
            if movie_title.lower() not in movie_titles_lower_set_without_duplicates:
                movie_titles_lower_set_without_duplicates.add(movie_title.lower())
                movie_titles_list_without_duplicates.append(movie_title)

        movie_titles_list_without_duplicates_upper_limit = list(movie_titles_list_without_duplicates.copy())
        assert (len(set(eval(str(movie_titles_list).lower()))) == len(movie_titles_list_without_duplicates))

        if len(movie_titles_list_without_duplicates_upper_limit) > 0:
            # if more than 10 movie titles --> use randomly 10 and keep ordering:
            while len(movie_titles_list_without_duplicates_upper_limit) > NR_RECOMMENDATIONS:
                movie_titles_list_without_duplicates_upper_limit.pop(
                    randint(0, len(movie_titles_list_without_duplicates_upper_limit) - 1))

        df.loc[len(df)] = [
            model,
            shot,
            cycle,
            filename.split("__")[0],  # submission_id
            valid_json,
            len(movie_titles_list),
            movie_titles_list,
            len(movie_titles_list_without_duplicates),
            movie_titles_list_without_duplicates,
            len(movie_titles_list_without_duplicates_upper_limit),
            movie_titles_list_without_duplicates_upper_limit,
        ]
    return df


def evaluateResultDirectory(directory, df):
    debug_msg("Evaluate directory " + str(directory), DebugLevel.DEBUG)
    cycle_directories = list()
    for cycle_directory_prefix in CYCLE_DIRECTORY_PREFIXES:
        cycle_directories += getAllDirectories(directory=directory, prefix=cycle_directory_prefix)
    for d in cycle_directories:
        _, model, shot, cycle = d.split("__")
        cycle = int(cycle.split("_")[-1])
        debug_msg("directory: %s, model: %s, shot: %s, cycle: %d" % (d, model, shot, cycle), DebugLevel.DEBUG)
        filename2content = readResponseFiles(directory=os.path.join(directory, d), suffix=RESPONSE_FILES_SUFFIX)
        filename2jsonResponses = prepareContent(filename2content)
        df = evaluateCycle(model, shot, cycle, filename2jsonResponses, df)
    return df


def extractRecommendations():
    df = pd.DataFrame(columns=[
        "model",
        "shot",
        "cycle",
        "submission_id",
        "valid_json",
        "nr_recs",
        "rec_list",
        "nr_recs_unique",
        "rec_list_unique",
        "nr_recs_unique_upper_limit",
        "rec_list_unique_upper_limit",
    ])
    result_directories = list()
    for results_directory_prefix in RESULTS_DIRECTORY_PREFIXES:
        result_directories += getAllDirectories(directory=RESULTS_DIRECTORY, prefix=results_directory_prefix)
    for result_directory in result_directories:
        df = evaluateResultDirectory(os.path.join(RESULTS_DIRECTORY, result_directory), df)
    df.to_csv(os.path.join(DIRECTORY, EXTRACTION_RESULTS_FILENAME), index=False)
    return df


def evaluateRecommendations(data_directory=DATA_DIRECTORY, multi_language=False):
    if not multi_language:
        submissionid2trueMovieTitles = loadPickle(data_directory + "/" + SUBMISSIONID_TRUEMOVIETITLES_FILENAME)
    else:
        submissionid2trueMovieTitles = loadPickle(
            data_directory + "/" + SUBMISSIONID_TRUEMOVIETITLES_MULTILANGUAGE_FILENAME)

    df_extraction = pd.read_csv(os.path.join(DIRECTORY, EXTRACTION_RESULTS_FILENAME))
    df_evaluation = pd.DataFrame(columns=[
        "model",
        "shot",
        "cycle",
        "submission_id",
        "valid_json",
        "nr_recs",
        "nr_recs_unique",
        "nr_recs_unique_upper_limit",
        "nr_true_recs",
        "exact_match_precision",
        "exact_match_recall",
        "exact_match_f1_score",
        "exact_match_nDCG",
        "threshold_match_precision",
        "threshold_match_recall",
        "threshold_match_f1_score",
        "threshold_match_nDCG",
    ])

    for index, row in df_extraction.iterrows():
        # exact match (case-insensitive):
        movie_titles_list_without_duplicates_upper_limit = eval(row["rec_list_unique_upper_limit"].lower())
        true_movie_titles_list = eval(str(submissionid2trueMovieTitles[row["submission_id"]]).lower())

        assert (type(movie_titles_list_without_duplicates_upper_limit) is list)
        assert (len(movie_titles_list_without_duplicates_upper_limit) == row["nr_recs_unique_upper_limit"])
        assert (type(true_movie_titles_list) is list)
        assert (len(true_movie_titles_list) == len(submissionid2trueMovieTitles[row["submission_id"]]))

        if not multi_language:
            exact_match_precision, exact_match_recall, exact_match_f1_score, exact_match_nDCG = prediction_metrics(
                true_movie_titles_list, movie_titles_list_without_duplicates_upper_limit)
            # match with similarity threshold (case-insensitive):
            threshold_match_precision, threshold_match_recall, threshold_match_f1_score, threshold_match_nDCG = prediction_metrics(
                true_movie_titles_list, movie_titles_list_without_duplicates_upper_limit, threshold=0.9)
        else:
            exact_match_precision, exact_match_recall, exact_match_f1_score, exact_match_nDCG = prediction_metrics_multi_language(
                true_movie_titles_list, movie_titles_list_without_duplicates_upper_limit)
            # match with similarity threshold (case-insensitive):
            threshold_match_precision, threshold_match_recall, threshold_match_f1_score, threshold_match_nDCG = prediction_metrics_multi_language(
                true_movie_titles_list, movie_titles_list_without_duplicates_upper_limit, threshold=0.9)

        df_evaluation.loc[len(df_evaluation)] = [
            row["model"],
            row["shot"],
            row["cycle"],
            row["submission_id"],
            row["valid_json"],
            row["nr_recs"],
            row["nr_recs_unique"],
            row["nr_recs_unique_upper_limit"],
            len(true_movie_titles_list),
            exact_match_precision,
            exact_match_recall,
            exact_match_f1_score,
            exact_match_nDCG,
            threshold_match_precision,
            threshold_match_recall,
            threshold_match_f1_score,
            threshold_match_nDCG,
        ]
    if not multi_language:
        df_evaluation.to_csv(os.path.join(DIRECTORY, EVALUATION_RESULTS_FILENAME), index=False)
    else:
        df_evaluation.to_csv(os.path.join(DIRECTORY, EVALUATION_RESULTS_MULTI_LANGUAGE_FILENAME), index=False)
    return


def bootstrapCIV6(d, nr_samples=100000, alpha=0.95):
    d = 1.0 * np.array(d)
    result = bs.bootstrap(d, num_iterations=nr_samples, stat_func=bs_stats.mean, alpha=1 - alpha)
    return result.lower_bound, result.upper_bound


def getMedianCycles(target_column, multi_language=False):
    df_extraction = pd.read_csv(os.path.join(DIRECTORY, EXTRACTION_RESULTS_FILENAME))
    if not multi_language:
        df_evaluation = pd.read_csv(os.path.join(DIRECTORY, EVALUATION_RESULTS_FILENAME))
    else:
        df_evaluation = pd.read_csv(os.path.join(DIRECTORY, EVALUATION_RESULTS_MULTI_LANGUAGE_FILENAME))

    df_extraction_median_cycle_list = list()
    df_evaluation_median_cycle_list = list()
    model_list = list(df_evaluation["model"].unique())
    for model in model_list:
        df_evaluation_model = df_evaluation.loc[df_evaluation["model"] == model].reset_index(drop=True)
        shot_list = list(df_evaluation_model["shot"].unique())
        for shot in shot_list:
            df_evaluation_model_shot = df_evaluation_model.loc[df_evaluation_model["shot"] == shot].reset_index(
                drop=True)
            cycle2value_mean = dict()
            for cycle in df_evaluation_model_shot["cycle"].unique():
                df_evaluation_model_shot_cycle = df_evaluation_model_shot.loc[
                    df_evaluation_model_shot["cycle"] == cycle].reset_index(drop=True)
                cycle2value_mean[cycle] = df_evaluation_model_shot_cycle[target_column].mean()

            # cycle with median mean f1 score:
            sorted_values_list = sorted(cycle2value_mean.values())
            median_cycle_value = sorted(sorted_values_list)[int(len(sorted_values_list) / 2)]
            for cycle, value in cycle2value_mean.items():
                if median_cycle_value == value:
                    break
            median_cycle = cycle
            df_evaluation_model_shot_mediancycle = df_evaluation_model_shot.loc[
                df_evaluation_model_shot["cycle"] == median_cycle]
            df_evaluation_median_cycle_list.append(df_evaluation_model_shot_mediancycle)
            df_extraction_model_shot_mediancycle = df_extraction.loc[
                (df_extraction["model"] == model) &
                (df_extraction["shot"] == shot) &
                (df_extraction["cycle"] == median_cycle)
                ]
            df_extraction_median_cycle_list.append(df_extraction_model_shot_mediancycle)

    df_extraction_median_cycle = pd.concat(df_extraction_median_cycle_list)
    df_evaluation_median_cycle = pd.concat(df_evaluation_median_cycle_list)

    df_extraction_median_cycle.to_csv(os.path.join(DIRECTORY, EXTRACTION_RESULTS_MEDIAN_CYCLES_FILENAME), index=False)
    if not multi_language:
        df_evaluation_median_cycle.to_csv(os.path.join(DIRECTORY, EVALUATION_RESULTS_MEDIAN_CYCLES_FILENAME),
                                          index=False)
    else:
        df_evaluation_median_cycle.to_csv(
            os.path.join(DIRECTORY, EVALUATION_RESULTS_MULTI_LANGUAGE_MEDIAN_CYCLES_FILENAME), index=False)
    return


def computeANOVA(target_column, multi_language=False):
    if not multi_language:
        df_evaluation = pd.read_csv(os.path.join(DIRECTORY, EVALUATION_RESULTS_FILENAME))
    else:
        df_evaluation = pd.read_csv(os.path.join(DIRECTORY, EVALUATION_RESULTS_MULTI_LANGUAGE_FILENAME))

    anova_latex_string = ""
    model_list = list(df_evaluation["model"].unique())
    for model in model_list:
        df_evaluation_model = df_evaluation.loc[df_evaluation["model"] == model].reset_index(drop=True)
        shot_list = list(df_evaluation_model["shot"].unique())
        for shot in shot_list:
            df_evaluation_model_shot = df_evaluation_model.loc[df_evaluation_model["shot"] == shot].reset_index(
                drop=True)
            anova_precision = AnovaRM(data=df_evaluation_model_shot,
                                      depvar=target_column.replace("f1_score", "precision"), subject='submission_id',
                                      within=['cycle']).fit()
            anova_recall = AnovaRM(data=df_evaluation_model_shot, depvar=target_column.replace("f1_score", "recall"),
                                   subject='submission_id', within=['cycle']).fit()
            anova_f1 = AnovaRM(data=df_evaluation_model_shot, depvar=target_column, subject='submission_id',
                               within=['cycle']).fit()
            anova_ndcg = AnovaRM(data=df_evaluation_model_shot, depvar=target_column.replace("f1_score", "nDCG"),
                                 subject='submission_id', within=['cycle']).fit()

            anova_latex_string += correctModelLabel(model)
            anova_latex_string += (" & %.4f" * 2 * 4) % (
                anova_precision.anova_table.values.tolist()[0][0],
                anova_precision.anova_table.values.tolist()[0][3],
                anova_recall.anova_table.values.tolist()[0][0],
                anova_recall.anova_table.values.tolist()[0][3],
                anova_f1.anova_table.values.tolist()[0][0],
                anova_f1.anova_table.values.tolist()[0][3],
                anova_ndcg.anova_table.values.tolist()[0][0],
                anova_ndcg.anova_table.values.tolist()[0][3],
            )
            anova_latex_string += " \\\\\n"
    print(anova_latex_string)
    return df_evaluation


def correctModelLabel(label):
    return (label
            .replace("-instruct", "")
            .replace("b-mini", "b")
            .replace("-medium", "")
            .replace("-q4_0", "")
            .replace("-v1.5", "")
            .replace("-v1.1", "")
            .replace("-v0.3", "")
            .replace("-v0.1", "")
            .replace("-2407", "")
            .replace("-2409", "")
            .replace("-0125", "")
            .replace("-2024-08-06", "")
            .replace("-2024-07-18", "")
            .replace("-chat", "")
            .replace("-128k", "")
            .replace("gemma2", "Gemma 2")
            .replace("gemma", "Gemma")
            .replace("gpt", "GPT")
            .replace("llama3", "Llama 3")
            .replace("mistral", "Mistral")
            .replace("mixtral", "Mixtral")
            .replace("phi3", "Phi-3")
            .replace("qwen2", "Qwen2")
            .replace("yi", "Yi")
            .replace("-mini", " mini")
            .replace("-nemo", " NeMo")
            .replace("-small", " Small")
            .replace("-large", " Large")
            .replace(":", " ")
            .replace("b", "B")
            .replace("-turBo", " Turbo")
            )


def formatterWithoutLeadingZeros(x, pos):
    """Format 1 as 1, 0 as 0, and all values whose absolute values is between
    0 and 1 without the leading "0." (e.g., 0.7 is formatted as .7 and -0.4 is
    formatted as -.4)."""
    val_str = '{:g}'.format(x)
    if np.abs(x) > 0 and np.abs(x) < 1:
        return val_str.replace("0", "", 1)
    else:
        return val_str


def createAccuracyHeatmapGrouped(model_group_list, shot_list, target_column, multi_language=False,
                                 values_in_percentages=False):
    if not multi_language:
        df_evaluation = pd.read_csv(os.path.join(DIRECTORY, EVALUATION_RESULTS_MEDIAN_CYCLES_FILENAME))
    else:
        df_evaluation = pd.read_csv(os.path.join(DIRECTORY, EVALUATION_RESULTS_MULTI_LANGUAGE_MEDIAN_CYCLES_FILENAME))

    plt.close()
    fig, axes = plt.subplots(
        nrows=len(model_group_list) + 1,
        figsize=(10, 12),
        gridspec_kw={"height_ratios": [len(model_group) for model_group in model_group_list] + [0.3]})
    cm = sns.light_palette("seagreen", as_cmap=True)

    df_list = list()
    ci_dict = defaultdict(list)
    for i, model_group in enumerate(model_group_list):
        data = {
            "Model": model_group.copy(),
        }
        all_annotations = list()
        for shot, shot_label in shot_list:
            df_evaluation_shot = df_evaluation.loc[df_evaluation["shot"] == shot].reset_index(drop=True)
            shot_values_list = list()
            annotations_list = list()
            for model in model_group:
                df_evaluation_shot_model = df_evaluation_shot.loc[df_evaluation_shot["model"] == model].reset_index(
                    drop=True)
                if len(df_evaluation_shot_model) == 0:
                    shot_values_list.append(None)
                    annotations_list.append("")
                    continue
                # Bootstrapped CI:
                lower, upper = bootstrapCIV6(list(df_evaluation_shot_model[target_column]),
                                             nr_samples=NR_BOOTSTRAP_SAMPLES)
                ci_dict[shot_label + "_ci_lo"].append(lower)
                ci_dict[shot_label + "_ci_hi"].append(upper)
                shot_values_list.append(df_evaluation_shot_model[target_column].mean())
                if values_in_percentages:
                    annotations_list.append("%.2f%%\n$^{[%.2f\\%%, %.2f\\%%]}$" % (
                    100.0 * df_evaluation_shot_model[target_column].mean(), 100.0 * lower, 100 * upper))
                else:
                    annotations_list.append(
                        ("%.4f\n" % df_evaluation_shot_model[target_column].mean()).replace("0.", ".") + (
                                CI_FORMAT_STRING % (lower, upper)).replace("0.", ".\!").replace(", .",
                                                                                                ",\; ."))  # "\;" a thick space; "\!" a negative thin space
            data[shot_label] = shot_values_list
            all_annotations.append(annotations_list)

        # correct model labels:
        for index in range(len(data["Model"])):
            data["Model"][index] = correctModelLabel(data["Model"][index])
        if i == 0:
            ylabel = "Tiny"
        elif i == 1:
            ylabel = "Small"
        elif i == 2:
            ylabel = "Medium"
        elif i == 3:
            ylabel = "Large"
        data[ylabel] = data["Model"]
        del data["Model"]
        df = pd.DataFrame(data)
        df.set_index(ylabel, inplace=True)
        df_list.append(df)
        all_annotations = np.array(all_annotations)
        all_annotations = all_annotations.T
        sns.heatmap(df, annot=all_annotations, fmt='', cmap=cm, ax=axes[i], cbar=False,
                    annot_kws={"va": 'center_baseline'}, vmin=0, vmax=0.22)
        if i == 0:
            axes[i].tick_params(left=True, bottom=False, top=True, right=False, labelleft=True, labelbottom=False,
                                labeltop=True)
            axes[i].xaxis.tick_top()
            axes[i].xaxis.set_label_position('top')
        else:
            axes[i].tick_params(left=True, bottom=False, top=False, right=False, labelleft=True, labelbottom=False)
    if values_in_percentages:
        fig.colorbar(axes[i].get_children()[0], cax=axes[i + 1], orientation="horizontal",
                     format=mtick.PercentFormatter(1.0))
    else:
        fig.colorbar(axes[i].get_children()[0], cax=axes[i + 1], orientation="horizontal",
                     format=mtick.FuncFormatter(formatterWithoutLeadingZeros))

    plt.tight_layout()
    plt.subplots_adjust(left=0.18)
    axes[1].text(x=-0.2, y=0.13, s="F1@10", ha='center', va='center', transform=axes[1].transAxes, rotation=90,
                 fontsize=14)
    plt.savefig("result_plots/results_accuracy_grouped.pdf")

    df_list = pd.concat(df_list)
    # add ci columns to csv:
    offset = 1
    for index, (col, values) in enumerate(ci_dict.items()):
        df_list.insert(index + offset, col, values, True)
        if index % 2 != 0: offset += 1
    df_list.to_csv("result_plots/results_accuracy_grouped.csv")
    return


def createAccuracyHeatmapGroupedByShots(model_group_list, shot_group_list, target_column, multi_language=False,
                                        values_in_percentages=False):
    if not multi_language:
        df_evaluation = pd.read_csv(os.path.join(DIRECTORY, EVALUATION_RESULTS_MEDIAN_CYCLES_FILENAME))
    else:
        df_evaluation = pd.read_csv(os.path.join(DIRECTORY, EVALUATION_RESULTS_MULTI_LANGUAGE_MEDIAN_CYCLES_FILENAME))

    plt.close()
    fig, axes = plt.subplots(
        nrows=len(model_group_list) + 1,
        figsize=(6, 14),
        gridspec_kw={"height_ratios": [len(model_group) for model_group in model_group_list] + [0.3]})
    cm = sns.light_palette("seagreen", as_cmap=True)

    df_list = list()
    ci_dict = defaultdict(list)

    for i, model_group in enumerate(model_group_list):
        data = {
            "Model": model_group.copy(),
        }
        all_annotations = list()
        for shot_label, shot_list in shot_group_list:
            df_evaluation_shot = df_evaluation.loc[df_evaluation["shot"].isin(shot_list)].reset_index(drop=True)
            shot_values_list = list()
            annotations_list = list()
            for model in model_group:
                df_evaluation_shot_model = df_evaluation_shot.loc[df_evaluation_shot["model"] == model].reset_index(
                    drop=True)
                if len(df_evaluation_shot_model) == 0:
                    shot_values_list.append(None)
                    annotations_list.append("")
                    continue
                # Bootstrapped CI:
                lower, upper = bootstrapCIV6(list(df_evaluation_shot_model[target_column]),
                                             nr_samples=NR_BOOTSTRAP_SAMPLES)
                ci_dict[shot_label + "_ci_lo"].append(lower)
                ci_dict[shot_label + "_ci_hi"].append(upper)
                shot_values_list.append(df_evaluation_shot_model[target_column].mean())
                if values_in_percentages:
                    annotations_list.append("%.2f%%\n$^{[%.2f\\%%, %.2f\\%%]}$" % (
                    100.0 * df_evaluation_shot_model[target_column].mean(), 100.0 * lower, 100 * upper))
                else:
                    annotations_list.append(
                        ("%.4f\n" % df_evaluation_shot_model[target_column].mean()).replace("0.", ".") + (
                                CI_FORMAT_STRING % (lower, upper)).replace("0.", ".\!").replace(", .",
                                                                                                ",\; ."))  # "\;" a thick space; "\!" a negative thin space

            # model group average row:
            df_evaluation_shot_model_group = df_evaluation_shot.loc[
                df_evaluation_shot["model"].isin(model_group)].reset_index(drop=True)
            if len(df_evaluation_shot_model_group) == 0:
                shot_values_list.append(None)
                annotations_list.append("")
                continue
            # Bootstrapped CI:
            lower, upper = bootstrapCIV6(list(df_evaluation_shot_model_group[target_column]),
                                         nr_samples=NR_BOOTSTRAP_SAMPLES)
            ci_dict[shot_label + "_ci_lo"].append(lower)
            ci_dict[shot_label + "_ci_hi"].append(upper)
            shot_values_list.append(df_evaluation_shot_model_group[target_column].mean())
            if values_in_percentages:
                annotations_list.append("%.2f%%\n$^{[%.2f\\%%, %.2f\\%%]}$" % (
                100.0 * df_evaluation_shot_model_group[target_column].mean(), 100.0 * lower, 100 * upper))
            else:
                annotations_list.append(
                    ("%.4f\n" % df_evaluation_shot_model_group[target_column].mean()).replace("0.", ".") + (
                            CI_FORMAT_STRING % (lower, upper)).replace("0.", ".\!").replace(", .",
                                                                                            ",\; ."))  # "\;" a thick space; "\!" a negative thin space

            data[shot_label] = shot_values_list
            all_annotations.append(annotations_list)

        # Average column:
        shot_values_list = list()
        annotations_list = list()
        for model in model_group:
            df_evaluation_model = df_evaluation.loc[df_evaluation["model"] == model].reset_index(drop=True)
            if len(df_evaluation_model) == 0:
                shot_values_list.append(None)
                annotations_list.append("")
                continue
            # Bootstrapped CI:
            lower, upper = bootstrapCIV6(list(df_evaluation_model[target_column]), nr_samples=NR_BOOTSTRAP_SAMPLES)
            ci_dict["Average_ci_lo"].append(lower)
            ci_dict["Average_ci_hi"].append(upper)

            shot_values_list.append(df_evaluation_model[target_column].mean())
            if values_in_percentages:
                annotations_list.append("%.2f%%\n$^{[%.2f\\%%, %.2f\\%%]}$" % (
                100.0 * df_evaluation_model[target_column].mean(), 100.0 * lower, 100 * upper))
            else:
                annotations_list.append(("%.4f\n" % df_evaluation_model[target_column].mean()).replace("0.", ".") + (
                        CI_FORMAT_STRING % (lower, upper)).replace("0.", ".\!").replace(", .",
                                                                                        ",\; ."))  # "\;" a thick space; "\!" a negative thin space

        # model group average row:
        df_evaluation_model_group = df_evaluation.loc[df_evaluation["model"].isin(model_group)].reset_index(drop=True)
        if len(df_evaluation_model_group) == 0:
            shot_values_list.append(None)
            annotations_list.append("")
            continue
        # Bootstrapped CI:
        lower, upper = bootstrapCIV6(list(df_evaluation_model_group[target_column]), nr_samples=NR_BOOTSTRAP_SAMPLES)
        ci_dict["Average_ci_lo"].append(lower)
        ci_dict["Average_ci_hi"].append(upper)
        shot_values_list.append(df_evaluation_model_group[target_column].mean())
        if values_in_percentages:
            annotations_list.append("%.2f%%\n$^{[%.2f\\%%, %.2f\\%%]}$" % (
            100.0 * df_evaluation_model_group[target_column].mean(), 100.0 * lower, 100 * upper))
        else:
            annotations_list.append(("%.4f\n" % df_evaluation_model_group[target_column].mean()).replace("0.", ".") + (
                    CI_FORMAT_STRING % (lower, upper)).replace("0.", ".\!").replace(", .",
                                                                                    ",\; ."))  # "\;" a thick space; "\!" a negative thin space

        data["Average"] = shot_values_list
        all_annotations.append(annotations_list)

        # correct model labels:
        for index in range(len(data["Model"])):
            data["Model"][index] = correctModelLabel(data["Model"][index])

        if i == 0:
            ylabel = "Tiny"
        elif i == 1:
            ylabel = "Small"
        elif i == 2:
            ylabel = "Medium"
        elif i == 3:
            ylabel = "Large"
        data[ylabel] = data["Model"]
        del data["Model"]
        data[ylabel].append("Average")
        df = pd.DataFrame(data)
        df.set_index(ylabel, inplace=True)
        df_list.append(df)
        all_annotations = np.array(all_annotations)
        all_annotations = all_annotations.T
        sns.heatmap(df, annot=all_annotations, fmt='', cmap=cm, ax=axes[i], cbar=False,
                    annot_kws={"va": 'center_baseline'}, vmin=0, vmax=0.22)
        if i == 0:
            axes[i].tick_params(left=True, bottom=False, top=True, right=False, labelleft=True, labelbottom=False,
                                labeltop=True)
            axes[i].xaxis.tick_top()
            axes[i].xaxis.set_label_position('top')
        else:
            axes[i].tick_params(left=True, bottom=False, top=False, right=False, labelleft=True, labelbottom=False)
    if values_in_percentages:
        fig.colorbar(axes[i].get_children()[0], cax=axes[i + 1], orientation="horizontal",
                     format=mtick.PercentFormatter(1.0))
    else:
        fig.colorbar(axes[i].get_children()[0], cax=axes[i + 1], orientation="horizontal",
                     format=mtick.FuncFormatter(formatterWithoutLeadingZeros))

    plt.tight_layout()
    plt.savefig("result_plots/results_accuracy_grouped_by_shots__%s.pdf" % target_column)

    df_list = pd.concat(df_list)
    # add ci columns to csv:
    offset = 1
    for index, (col, values) in enumerate(ci_dict.items()):
        df_list.insert(index + offset, col, values, True)
        if index % 2 != 0: offset += 1
    df_list.to_csv("result_plots/results_accuracy_grouped_by_shots__%s.csv" % target_column)
    return


def createAccuracyHeatmapGroupedByModelsAndShots(model_group_list, shot_group_list, target_column, multi_language=False,
                                                 values_in_percentages=False):
    if not multi_language:
        df_evaluation = pd.read_csv(os.path.join(DIRECTORY, EVALUATION_RESULTS_MEDIAN_CYCLES_FILENAME))
    else:
        df_evaluation = pd.read_csv(os.path.join(DIRECTORY, EVALUATION_RESULTS_MULTI_LANGUAGE_MEDIAN_CYCLES_FILENAME))

    plt.close()
    fig, axes = plt.subplots(
        nrows=2,
        figsize=(5, 2.2),
        gridspec_kw={"height_ratios": [2.2, 0.1]})
    cm = sns.light_palette("seagreen", as_cmap=True)

    ci_dict = defaultdict(list)
    data = {
        "Model Group": ["Tiny", "Small", "Medium", "Large"]
    }
    all_annotations = list()
    for shot_label, shot_list in shot_group_list:
        df_evaluation_shot = df_evaluation.loc[df_evaluation["shot"].isin(shot_list)].reset_index(drop=True)
        shot_values_list = list()
        annotations_list = list()
        for i, model_group in enumerate(model_group_list):
            # model group average row:
            df_evaluation_shot_model_group = df_evaluation_shot.loc[
                df_evaluation_shot["model"].isin(model_group)].reset_index(drop=True)
            if len(df_evaluation_shot_model_group) == 0:
                shot_values_list.append(None)
                annotations_list.append("")
                continue
            # Bootstrapped CI:
            lower, upper = bootstrapCIV6(list(df_evaluation_shot_model_group[target_column]),
                                         nr_samples=NR_BOOTSTRAP_SAMPLES)
            ci_dict[shot_label + "_ci_lo"].append(lower)
            ci_dict[shot_label + "_ci_hi"].append(upper)
            shot_values_list.append(df_evaluation_shot_model_group[target_column].mean())
            if values_in_percentages:
                annotations_list.append("%.2f%%\n$^{[%.2f\\%%, %.2f\\%%]}$" % (
                100.0 * df_evaluation_shot_model_group[target_column].mean(), 100.0 * lower, 100 * upper))
            else:
                annotations_list.append(
                    ("%.4f\n" % df_evaluation_shot_model_group[target_column].mean()).replace("0.", ".") + (
                            CI_FORMAT_STRING % (lower, upper)).replace("0.", ".\!").replace(", .",
                                                                                            ",\; ."))  # "\;" a thick space; "\!" a negative thin space

        data[shot_label] = shot_values_list
        all_annotations.append(annotations_list)

    # Average column:
    shot_values_list = list()
    annotations_list = list()
    for i, model_group in enumerate(model_group_list):
        # model group average row:
        df_evaluation_model_group = df_evaluation.loc[df_evaluation["model"].isin(model_group)].reset_index(drop=True)
        if len(df_evaluation_model_group) == 0:
            shot_values_list.append(None)
            annotations_list.append("")
            continue
        # Bootstrapped CI:
        lower, upper = bootstrapCIV6(list(df_evaluation_model_group[target_column]), nr_samples=NR_BOOTSTRAP_SAMPLES)
        ci_dict["Average_ci_lo"].append(lower)
        ci_dict["Average_ci_hi"].append(upper)
        shot_values_list.append(df_evaluation_model_group[target_column].mean())
        if values_in_percentages:
            annotations_list.append("%.2f%%\n$^{[%.2f\\%%, %.2f\\%%]}$" % (
            100.0 * df_evaluation_model_group[target_column].mean(), 100.0 * lower, 100 * upper))
        else:
            annotations_list.append(("%.4f\n" % df_evaluation_model_group[target_column].mean()).replace("0.", ".") + (
                    CI_FORMAT_STRING % (lower, upper)).replace("0.", ".\!").replace(", .",
                                                                                    ",\; ."))  # "\;" a thick space; "\!" a negative thin space

    data["Average"] = shot_values_list
    all_annotations.append(annotations_list)
    df = pd.DataFrame(data)
    df.set_index("Model Group", inplace=True)
    all_annotations = np.array(all_annotations)
    all_annotations = all_annotations.T
    sns.heatmap(df, annot=all_annotations, fmt='', cmap=cm, ax=axes[0], cbar=False, annot_kws={"va": 'center_baseline'},
                vmin=0, vmax=0.18)
    axes[0].set_ylabel("", rotation=0, labelpad=100)
    axes[0].tick_params(left=True, bottom=False, top=True, right=False, labelleft=True, labelbottom=False,
                        labeltop=True)
    axes[0].xaxis.tick_top()
    axes[0].xaxis.set_label_position('top')

    if values_in_percentages:
        fig.colorbar(axes[0].get_children()[0], cax=axes[1], orientation="horizontal",
                     format=mtick.PercentFormatter(1.0))
    else:
        fig.colorbar(axes[0].get_children()[0], cax=axes[1], orientation="horizontal",
                     format=mtick.FuncFormatter(formatterWithoutLeadingZeros))

    plt.tight_layout()
    plt.savefig("result_plots/results_accuracy_grouped_by_models_and_shots__%s.pdf" % target_column)

    # add ci columns to csv:
    offset = 1
    for index, (col, values) in enumerate(ci_dict.items()):
        df.insert(index + offset, col, values, True)
        if index % 2 != 0: offset += 1
    df.to_csv("result_plots/results_accuracy_grouped_by_models_and_shots__%s.csv" % target_column)
    return


def plotValidJsonGrouped(df_extraction, model_group_list, shot_list, values_in_percentages=False):
    plt.close()
    fig, axes = plt.subplots(
        nrows=len(model_group_list) + 1,
        figsize=(10, 12),
        gridspec_kw={"height_ratios": [len(model_group) for model_group in model_group_list] + [0.3]})
    cm = sns.light_palette("#79C", as_cmap=True)

    df_list = list()
    ci_dict = defaultdict(list)
    for i, model_group in enumerate(model_group_list):
        data = {
            "Model": model_group.copy(),
        }
        all_annotations = list()
        for shot, shot_label in shot_list:
            df_extraction_shot = df_extraction.loc[df_extraction["shot"] == shot].reset_index(drop=True)
            shot_values_list = list()
            annotations_list = list()
            for model in model_group:
                df_extraction_shot_model = df_extraction_shot.loc[df_extraction_shot["model"] == model].reset_index(
                    drop=True)
                assert len(df_extraction_shot_model.loc[df_extraction_shot_model['valid_json'] == True]) + len(
                    df_extraction_shot_model.loc[df_extraction_shot_model['valid_json'] == False]) == len(
                    df_extraction_shot_model)
                valid_json_percentage = len(
                    df_extraction_shot_model.loc[df_extraction_shot_model['valid_json'] == True]) / len(
                    df_extraction_shot_model)
                valid_json_list = (
                        [1] * len(df_extraction_shot_model.loc[df_extraction_shot_model['valid_json'] == True]) +
                        [0] * len(df_extraction_shot_model.loc[df_extraction_shot_model['valid_json'] == False]))
                assert valid_json_percentage == np.mean(valid_json_list)
                # Bootstrapped CI:
                lower, upper = bootstrapCIV6(list(valid_json_list), nr_samples=NR_BOOTSTRAP_SAMPLES)
                ci_dict[shot_label + "_ci_lo"].append(lower)
                ci_dict[shot_label + "_ci_hi"].append(upper)
                shot_values_list.append(valid_json_percentage)
                annotations_list.append(
                    ("%.4f\n" % valid_json_percentage).replace("0.", ".") + (CI_FORMAT_STRING % (lower, upper)).replace(
                        "0.", ".\!").replace(", .", ",\; ."))  # "\;" a thick space; "\!" a negative thin space

            data[shot_label] = shot_values_list
            all_annotations.append(annotations_list)

        # correct model labels:
        for index in range(len(data["Model"])):
            data["Model"][index] = correctModelLabel(data["Model"][index])
        if i == 0:
            ylabel = "Tiny"
        elif i == 1:
            ylabel = "Small"
        elif i == 2:
            ylabel = "Medium"
        elif i == 3:
            ylabel = "Large"
        data[ylabel] = data["Model"]
        del data["Model"]
        df = pd.DataFrame(data)
        df.set_index(ylabel, inplace=True)
        df_list.append(df)
        all_annotations = np.array(all_annotations)
        all_annotations = all_annotations.T
        if values_in_percentages:
            ax = sns.heatmap(df, annot=True, fmt='.2%', cmap=cm, ax=axes[i], cbar=False,
                             annot_kws={"va": 'center_baseline'}, vmin=0.68, vmax=1)
        else:
            ax = sns.heatmap(df, annot=all_annotations, fmt="", cmap=cm, ax=axes[i], cbar=False,
                             annot_kws={"va": 'center_baseline'}, vmin=0.68, vmax=1)
        if i == 0:
            axes[i].tick_params(left=True, bottom=False, top=True, right=False, labelleft=True, labelbottom=False,
                                labeltop=True)
            axes[i].xaxis.tick_top()
            axes[i].xaxis.set_label_position('top')
        else:
            axes[i].tick_params(left=True, bottom=False, top=False, right=False, labelleft=True, labelbottom=False)
    if values_in_percentages:
        fig.colorbar(axes[i].get_children()[0], cax=axes[i + 1], orientation="horizontal",
                     format=mtick.PercentFormatter(1.0))
    else:
        fig.colorbar(axes[i].get_children()[0], cax=axes[i + 1], orientation="horizontal",
                     format=mtick.FuncFormatter(formatterWithoutLeadingZeros))

    plt.tight_layout()
    plt.subplots_adjust(left=0.18)
    axes[1].text(x=-0.2, y=0.13, s="Valid JSON Format", ha='center', va='center', transform=axes[1].transAxes,
                 rotation=90, fontsize=14)
    plt.savefig("result_plots/results_valid_jsons_grouped.pdf")

    df_list = pd.concat(df_list)
    # add ci columns to csv:
    offset = 1
    for index, (col, values) in enumerate(ci_dict.items()):
        df_list.insert(index + offset, col, values, True)
        if index % 2 != 0: offset += 1
    df_list.to_csv("result_plots/results_valid_jsons_grouped.csv")
    return


def plotValidJsonGroupedByShots(df_extraction, model_group_list, shot_group_list, values_in_percentages=False):
    plt.close()
    fig, axes = plt.subplots(
        nrows=len(model_group_list) + 1,
        figsize=(6, 14),
        gridspec_kw={"height_ratios": [len(model_group) for model_group in model_group_list] + [0.3]})
    cm = sns.light_palette("#79C", as_cmap=True)

    df_list = list()
    ci_dict = defaultdict(list)
    for i, model_group in enumerate(model_group_list):
        data = {
            "Model": model_group.copy(),
        }
        all_annotations = list()
        for shot_label, shot_list in shot_group_list:
            df_extraction_shot = df_extraction.loc[df_extraction["shot"].isin(shot_list)].reset_index(drop=True)
            shot_values_list = list()
            annotations_list = list()
            for model in model_group:
                df_extraction_shot_model = df_extraction_shot.loc[df_extraction_shot["model"] == model].reset_index(
                    drop=True)
                assert len(df_extraction_shot_model.loc[df_extraction_shot_model['valid_json'] == True]) + len(
                    df_extraction_shot_model.loc[df_extraction_shot_model['valid_json'] == False]) == len(
                    df_extraction_shot_model)
                valid_json_percentage = len(
                    df_extraction_shot_model.loc[df_extraction_shot_model['valid_json'] == True]) / len(
                    df_extraction_shot_model)
                valid_json_list = (
                        [1] * len(df_extraction_shot_model.loc[df_extraction_shot_model['valid_json'] == True]) +
                        [0] * len(df_extraction_shot_model.loc[df_extraction_shot_model['valid_json'] == False]))
                assert valid_json_percentage == np.mean(valid_json_list)
                # Bootstrapped CI:
                lower, upper = bootstrapCIV6(list(valid_json_list), nr_samples=NR_BOOTSTRAP_SAMPLES)
                ci_dict[shot_label + "_ci_lo"].append(lower)
                ci_dict[shot_label + "_ci_hi"].append(upper)
                shot_values_list.append(valid_json_percentage)
                annotations_list.append(
                    ("%.4f\n" % valid_json_percentage).replace("0.", ".") + (CI_FORMAT_STRING % (lower, upper)).replace(
                        "0.", ".\!").replace(", .", ",\; ."))  # "\;" a thick space; "\!" a negative thin space

            # model group average row:
            df_extraction_shot_model_group = df_extraction_shot.loc[
                df_extraction_shot["model"].isin(model_group)].reset_index(drop=True)
            if len(df_extraction_shot_model_group) == 0:
                shot_values_list.append(None)
                annotations_list.append("")
                continue
            assert len(df_extraction_shot_model_group.loc[df_extraction_shot_model_group['valid_json'] == True]) + len(
                df_extraction_shot_model_group.loc[df_extraction_shot_model_group['valid_json'] == False]) == len(
                df_extraction_shot_model_group)
            valid_json_percentage = len(
                df_extraction_shot_model_group.loc[df_extraction_shot_model_group['valid_json'] == True]) / len(
                df_extraction_shot_model_group)
            valid_json_list = ([1] * len(
                df_extraction_shot_model_group.loc[df_extraction_shot_model_group['valid_json'] == True]) +
                               [0] * len(
                    df_extraction_shot_model_group.loc[df_extraction_shot_model_group['valid_json'] == False]))
            assert valid_json_percentage == np.mean(valid_json_list)
            # Bootstrapped CI:
            lower, upper = bootstrapCIV6(list(valid_json_list), nr_samples=NR_BOOTSTRAP_SAMPLES)
            ci_dict[shot_label + "_ci_lo"].append(lower)
            ci_dict[shot_label + "_ci_hi"].append(upper)
            shot_values_list.append(valid_json_percentage)
            annotations_list.append(
                ("%.4f\n" % valid_json_percentage).replace("0.", ".") + (CI_FORMAT_STRING % (lower, upper)).replace(
                    "0.", ".\!").replace(", .", ",\; ."))  # "\;" a thick space; "\!" a negative thin space

            data[shot_label] = shot_values_list
            all_annotations.append(annotations_list)

        # Average column:
        shot_values_list = list()
        annotations_list = list()
        for model in model_group:
            df_extraction_model = df_extraction.loc[df_extraction["model"] == model].reset_index(drop=True)
            assert len(df_extraction_model.loc[df_extraction_model['valid_json'] == True]) + len(
                df_extraction_model.loc[df_extraction_model['valid_json'] == False]) == len(df_extraction_model)
            valid_json_percentage = len(df_extraction_model.loc[df_extraction_model['valid_json'] == True]) / len(
                df_extraction_model)
            valid_json_list = ([1] * len(df_extraction_model.loc[df_extraction_model['valid_json'] == True]) +
                               [0] * len(df_extraction_model.loc[df_extraction_model['valid_json'] == False]))
            assert valid_json_percentage == np.mean(valid_json_list)
            # Bootstrapped CI:
            lower, upper = bootstrapCIV6(list(valid_json_list), nr_samples=NR_BOOTSTRAP_SAMPLES)
            ci_dict["Average_ci_lo"].append(lower)
            ci_dict["Average_ci_hi"].append(upper)
            shot_values_list.append(valid_json_percentage)
            annotations_list.append(
                ("%.4f\n" % valid_json_percentage).replace("0.", ".") + (CI_FORMAT_STRING % (lower, upper)).replace(
                    "0.", ".\!").replace(", .", ",\; ."))  # "\;" a thick space; "\!" a negative thin space

        # model group average row:
        df_extraction_model_group = df_extraction.loc[df_extraction["model"].isin(model_group)].reset_index(drop=True)
        if len(df_extraction_model_group) == 0:
            shot_values_list.append(None)
            annotations_list.append("")
            continue
        assert len(df_extraction_model_group.loc[df_extraction_model_group['valid_json'] == True]) + len(
            df_extraction_model_group.loc[df_extraction_model_group['valid_json'] == False]) == len(
            df_extraction_model_group)
        valid_json_percentage = len(
            df_extraction_model_group.loc[df_extraction_model_group['valid_json'] == True]) / len(
            df_extraction_model_group)
        valid_json_list = ([1] * len(df_extraction_model_group.loc[df_extraction_model_group['valid_json'] == True]) +
                           [0] * len(df_extraction_model_group.loc[df_extraction_model_group['valid_json'] == False]))
        assert valid_json_percentage == np.mean(valid_json_list)
        # Bootstrapped CI:
        lower, upper = bootstrapCIV6(list(valid_json_list), nr_samples=NR_BOOTSTRAP_SAMPLES)
        ci_dict["Average_ci_lo"].append(lower)
        ci_dict["Average_ci_hi"].append(upper)
        shot_values_list.append(valid_json_percentage)
        annotations_list.append(
            ("%.4f\n" % valid_json_percentage).replace("0.", ".") + (CI_FORMAT_STRING % (lower, upper)).replace("0.",
                                                                                                                ".\!").replace(
                ", .", ",\; ."))  # "\;" a thick space; "\!" a negative thin space
        data["Average"] = shot_values_list
        all_annotations.append(annotations_list)

        # correct model labels:
        for index in range(len(data["Model"])):
            data["Model"][index] = correctModelLabel(data["Model"][index])
        if i == 0:
            ylabel = "Tiny"
        elif i == 1:
            ylabel = "Small"
        elif i == 2:
            ylabel = "Medium"
        elif i == 3:
            ylabel = "Large"
        data[ylabel] = data["Model"]
        del data["Model"]
        data[ylabel].append("Average")
        df = pd.DataFrame(data)
        df.set_index(ylabel, inplace=True)
        df_list.append(df)
        all_annotations = np.array(all_annotations)
        all_annotations = all_annotations.T
        if values_in_percentages:
            ax = sns.heatmap(df, annot=True, fmt='.2%', cmap=cm, ax=axes[i], cbar=False,
                             annot_kws={"va": 'center_baseline'}, vmin=0.82, vmax=1)
        else:
            ax = sns.heatmap(df, annot=all_annotations, fmt="", cmap=cm, ax=axes[i], cbar=False,
                             annot_kws={"va": 'center_baseline'}, vmin=0.82, vmax=1)
        if i == 0:
            axes[i].tick_params(left=True, bottom=False, top=True, right=False, labelleft=True, labelbottom=False,
                                labeltop=True)
            axes[i].xaxis.tick_top()
            axes[i].xaxis.set_label_position('top')
        else:
            axes[i].tick_params(left=True, bottom=False, top=False, right=False, labelleft=True, labelbottom=False)
    if values_in_percentages:
        fig.colorbar(axes[i].get_children()[0], cax=axes[i + 1], orientation="horizontal",
                     format=mtick.PercentFormatter(1.0))
    else:
        fig.colorbar(axes[i].get_children()[0], cax=axes[i + 1], orientation="horizontal",
                     format=mtick.FuncFormatter(formatterWithoutLeadingZeros))

    plt.tight_layout()
    plt.savefig("result_plots/results_valid_jsons_grouped_by_shots.pdf")

    df_list = pd.concat(df_list)
    # add ci columns to csv:
    offset = 1
    for index, (col, values) in enumerate(ci_dict.items()):
        df_list.insert(index + offset, col, values, True)
        if index % 2 != 0: offset += 1
    df_list.to_csv("result_plots/results_valid_jsons_grouped_by_shots.csv")
    return


def plotValidJsonGroupedByModelsAndShots(df_extraction, model_group_list, shot_group_list, values_in_percentages=False):
    plt.close()
    fig, axes = plt.subplots(
        nrows=2,
        figsize=(5, 2.2),
        gridspec_kw={"height_ratios": [2.2, 0.1]})
    cm = sns.light_palette("#79C", as_cmap=True)

    ci_dict = defaultdict(list)
    data = {
        "Model Group": ["Tiny", "Small", "Medium", "Large"]
    }

    all_annotations = list()
    for shot_label, shot_list in shot_group_list:
        df_extraction_shot = df_extraction.loc[df_extraction["shot"].isin(shot_list)].reset_index(drop=True)
        shot_values_list = list()
        annotations_list = list()
        for i, model_group in enumerate(model_group_list):
            # model group average row:
            df_extraction_shot_model_group = df_extraction_shot.loc[
                df_extraction_shot["model"].isin(model_group)].reset_index(drop=True)
            if len(df_extraction_shot_model_group) == 0:
                shot_values_list.append(None)
                annotations_list.append("")
                continue
            assert len(df_extraction_shot_model_group.loc[df_extraction_shot_model_group['valid_json'] == True]) + len(
                df_extraction_shot_model_group.loc[df_extraction_shot_model_group['valid_json'] == False]) == len(
                df_extraction_shot_model_group)
            valid_json_percentage = len(
                df_extraction_shot_model_group.loc[df_extraction_shot_model_group['valid_json'] == True]) / len(
                df_extraction_shot_model_group)
            valid_json_list = ([1] * len(
                df_extraction_shot_model_group.loc[df_extraction_shot_model_group['valid_json'] == True]) +
                               [0] * len(
                    df_extraction_shot_model_group.loc[df_extraction_shot_model_group['valid_json'] == False]))
            assert valid_json_percentage == np.mean(valid_json_list)
            # Bootstrapped CI:
            lower, upper = bootstrapCIV6(list(valid_json_list), nr_samples=NR_BOOTSTRAP_SAMPLES)
            ci_dict[shot_label + "_ci_lo"].append(lower)
            ci_dict[shot_label + "_ci_hi"].append(upper)
            shot_values_list.append(valid_json_percentage)
            annotations_list.append(
                ("%.4f\n" % valid_json_percentage).replace("0.", ".") + (CI_FORMAT_STRING % (lower, upper)).replace(
                    "0.", ".\!").replace(", .", ",\; ."))  # "\;" a thick space; "\!" a negative thin space

        # Average row:
        data[shot_label] = shot_values_list
        all_annotations.append(annotations_list)

    # Average column:
    shot_values_list = list()
    annotations_list = list()
    for i, model_group in enumerate(model_group_list):
        # model group average row:
        df_extraction_model_group = df_extraction.loc[df_extraction["model"].isin(model_group)].reset_index(drop=True)
        if len(df_extraction_model_group) == 0:
            shot_values_list.append(None)
            annotations_list.append("")
            continue
        assert len(df_extraction_model_group.loc[df_extraction_model_group['valid_json'] == True]) + len(
            df_extraction_model_group.loc[df_extraction_model_group['valid_json'] == False]) == len(
            df_extraction_model_group)
        valid_json_percentage = len(
            df_extraction_model_group.loc[df_extraction_model_group['valid_json'] == True]) / len(
            df_extraction_model_group)
        valid_json_list = ([1] * len(df_extraction_model_group.loc[df_extraction_model_group['valid_json'] == True]) +
                           [0] * len(df_extraction_model_group.loc[df_extraction_model_group['valid_json'] == False]))
        assert valid_json_percentage == np.mean(valid_json_list)
        # Bootstrapped CI:
        lower, upper = bootstrapCIV6(list(valid_json_list), nr_samples=NR_BOOTSTRAP_SAMPLES)
        ci_dict["Average_ci_lo"].append(lower)
        ci_dict["Average_ci_hi"].append(upper)
        shot_values_list.append(valid_json_percentage)
        annotations_list.append(
            ("%.4f\n" % valid_json_percentage).replace("0.", ".") + (CI_FORMAT_STRING % (lower, upper)).replace("0.",
                                                                                                                ".\!").replace(
                ", .", ",\; ."))  # "\;" a thick space; "\!" a negative thin space
    # Average row:
    data["Average"] = shot_values_list
    all_annotations.append(annotations_list)
    df = pd.DataFrame(data)
    df.set_index("Model Group", inplace=True)
    all_annotations = np.array(all_annotations)
    all_annotations = all_annotations.T
    sns.heatmap(df, annot=all_annotations, fmt='', cmap=cm, ax=axes[0], cbar=False, annot_kws={"va": 'center_baseline'},
                vmin=0.97, vmax=1)
    axes[0].set_ylabel("", rotation=0, labelpad=100)
    axes[0].tick_params(left=True, bottom=False, top=True, right=False, labelleft=True, labelbottom=False,
                        labeltop=True)
    axes[0].xaxis.tick_top()
    axes[0].xaxis.set_label_position('top')
    if values_in_percentages:
        fig.colorbar(axes[0].get_children()[0], cax=axes[1], orientation="horizontal",
                     format=mtick.PercentFormatter(1.0))
    else:
        fig.colorbar(axes[0].get_children()[0], cax=axes[1], orientation="horizontal",
                     format=mtick.FuncFormatter(formatterWithoutLeadingZeros))

    plt.tight_layout()
    plt.savefig("result_plots/results_valid_jsons_grouped_by_models_and_shots.pdf")

    # add ci columns to csv:
    offset = 1
    for index, (col, values) in enumerate(ci_dict.items()):
        df.insert(index + offset, col, values, True)
        if index % 2 != 0: offset += 1
    df.to_csv("result_plots/results_valid_jsons_grouped_by_models_and_shots.csv")
    return


def plotNrRecsHeatmapGrouped(df_extraction, model_group_list, shot_list, values_in_percentages=False):
    plt.close()
    fig, axes = plt.subplots(
        nrows=len(model_group_list) + 2,
        ncols=len(shot_list),
        figsize=(21, 18), sharex=False, sharey=False,
        gridspec_kw={"height_ratios": [len(model_group) for model_group in model_group_list] + [0.3, 0.3]}
    )
    cm = sns.light_palette("gold", as_cmap=True)

    df_list = list()
    ci_dict = defaultdict(list)
    for j, (shot, shot_label) in enumerate(shot_list):
        df_extraction_shot = df_extraction.loc[df_extraction["shot"] == shot].reset_index(drop=True)
        df_list_shot = list()
        if len(df_extraction_shot['nr_recs']) == 0: continue
        for i, model_group in enumerate(model_group_list):
            all_annotations = [list(), list(), list(), list()]
            data = defaultdict(list)
            data["Model"] = model_group.copy()
            for k, model in enumerate(model_group):
                df_extraction_shot_model = df_extraction_shot.loc[df_extraction_shot["model"] == model].reset_index(
                    drop=True)

                data["0"].append(len(df_extraction_shot_model.loc[df_extraction_shot_model['nr_recs'] == 0]) / len(
                    df_extraction_shot_model))
                data["1-9"].append(len(df_extraction_shot_model.loc[(df_extraction_shot_model['nr_recs'] > 0) & (
                        df_extraction_shot_model['nr_recs'] < 10)]) / len(df_extraction_shot_model))
                data["10"].append(len(df_extraction_shot_model.loc[df_extraction_shot_model['nr_recs'] == 10]) / len(
                    df_extraction_shot_model))
                data[">10"].append(len(df_extraction_shot_model.loc[df_extraction_shot_model['nr_recs'] > 10]) / len(
                    df_extraction_shot_model))

                # Bootstrapped CI:
                nr_recs_only_0_list = (
                    [1] * len(df_extraction_shot_model.loc[df_extraction_shot_model['nr_recs'] == 0]) +
                    [0] * len(df_extraction_shot_model.loc[df_extraction_shot_model['nr_recs'] != 0]))
                lower, upper = bootstrapCIV6(list(nr_recs_only_0_list), nr_samples=NR_BOOTSTRAP_SAMPLES)
                ci_dict[shot_label + "_0_ci_lo"].append(lower)
                ci_dict[shot_label + "_0_ci_hi"].append(upper)
                all_annotations[0].append(
                    ("%.4f\n" % data["0"][-1]).replace("0.", ".") + (CI_FORMAT_STRING % (lower, upper)).replace("0.",
                                                                                                                ".\!").replace(
                        ", .", ",\; ."))  # "\;" a thick space; "\!" a negative thin space

                nr_recs_only_1to9_list = (
                    [1] * len(df_extraction_shot_model.loc[(df_extraction_shot_model['nr_recs'] > 0) & (
                        df_extraction_shot_model['nr_recs'] < 10)]) +
                    [0] * len(df_extraction_shot_model.loc[(df_extraction_shot_model['nr_recs'] == 0) | (
                        df_extraction_shot_model['nr_recs'] >= 10)]))
                lower, upper = bootstrapCIV6(list(nr_recs_only_1to9_list), nr_samples=NR_BOOTSTRAP_SAMPLES)
                ci_dict[shot_label + "_1-9_ci_lo"].append(lower)
                ci_dict[shot_label + "_1-9_ci_hi"].append(upper)
                all_annotations[1].append(
                    ("%.4f\n" % data["1-9"][-1]).replace("0.", ".") + (CI_FORMAT_STRING % (lower, upper)).replace("0.",
                                                                                                                  ".\!").replace(
                        ", .", ",\; ."))  # "\;" a thick space; "\!" a negative thin space

                nr_recs_only_10_list = (
                    [1] * len(df_extraction_shot_model.loc[df_extraction_shot_model['nr_recs'] == 10]) +
                    [0] * len(df_extraction_shot_model.loc[df_extraction_shot_model['nr_recs'] != 10]))
                lower, upper = bootstrapCIV6(list(nr_recs_only_10_list), nr_samples=NR_BOOTSTRAP_SAMPLES)
                ci_dict[shot_label + "_10_ci_lo"].append(lower)
                ci_dict[shot_label + "_10_ci_hi"].append(upper)
                all_annotations[2].append(
                    ("%.4f\n" % data["10"][-1]).replace("0.", ".") + (CI_FORMAT_STRING % (lower, upper)).replace("0.",
                                                                                                                 ".\!").replace(
                        ", .", ",\; ."))  # "\;" a thick space; "\!" a negative thin space

                nr_recs_only_greater10_list = (
                    [1] * len(df_extraction_shot_model.loc[df_extraction_shot_model['nr_recs'] > 10]) +
                    [0] * len(df_extraction_shot_model.loc[df_extraction_shot_model['nr_recs'] <= 10]))
                lower, upper = bootstrapCIV6(list(nr_recs_only_greater10_list), nr_samples=NR_BOOTSTRAP_SAMPLES)
                ci_dict[shot_label + "_>10_ci_lo"].append(lower)
                ci_dict[shot_label + "_>10_ci_hi"].append(upper)
                all_annotations[3].append(
                    ("%.4f\n" % data[">10"][-1]).replace("0.", ".") + (CI_FORMAT_STRING % (lower, upper)).replace("0.",
                                                                                                                  ".\!").replace(
                        ", .", ",\; ."))  # "\;" a thick space; "\!" a negative thin space

            data = dict(data)
            # correct model labels:
            for index in range(len(data["Model"])):
                data["Model"][index] = correctModelLabel(data["Model"][index])
            if i == 0:
                ylabel = "Tiny"
            elif i == 1:
                ylabel = "Small"
            elif i == 2:
                ylabel = "Medium"
            elif i == 3:
                ylabel = "Large"
            data[ylabel] = data["Model"]
            del data["Model"]
            df = pd.DataFrame(data)
            df.set_index(ylabel, inplace=True)
            df_list_shot.append(df.rename(
                columns={"0": shot_label + "_0", "1-9": shot_label + "_1-9", "10": shot_label + "_10",
                         ">10": shot_label + "_>10"}))
            all_annotations = np.array(all_annotations)
            all_annotations = all_annotations.T

            if j != len(shot_list) + 1 - 1:
                sns.heatmap(df, annot=all_annotations, fmt="", cmap=cm, ax=axes[i, j], cbar=False,
                            annot_kws={"va": 'center_baseline'}, vmin=0, vmax=1)
            else:
                sns.heatmap(df, annot=all_annotations, fmt="", cmap=cm, ax=axes[i, j],
                            annot_kws={"va": 'center_baseline'}, vmin=0, vmax=1)

            left = False
            bottom = False
            if j == 0:
                left = True
                axes[i, j].get_yaxis().label.set_rotation(270)
                axes[i, j].yaxis.set_label_coords(-0.52, 0.5)
            else:
                axes[i, j].set(xlabel="", ylabel="")
            if i == 0:
                axes[i, j].text(x=0.5, y=1.01, s=shot_label.replace("\n", " "), ha='center', va='bottom',
                                transform=axes[i, j].transAxes)
            if i == len(model_group_list) - 1:
                bottom = True
                axes[i, j].xaxis.tick_bottom()
                axes[i, j].xaxis.set_label_position('top')
            axes[i, j].tick_params(bottom=bottom, top=False, right=False, labelbottom=bottom, labeltop=False, left=left,
                                   labelleft=left)
        df_list.append(pd.concat(df_list_shot))

    gs = axes[-2, 0].get_gridspec()
    # remove the underlying Axes
    for ax in axes[-2, :]:
        ax.remove()
    for ax in axes[-1, :]:
        ax.remove()
    axbig = fig.add_subplot(gs[-1, :])
    if values_in_percentages:
        fig.colorbar(axes[i, j].get_children()[0], cax=axbig, orientation="horizontal",
                     format=mtick.PercentFormatter(1.0))
    else:
        fig.colorbar(axes[i, j].get_children()[0], cax=axbig, orientation="horizontal",
                     format=mtick.FuncFormatter(formatterWithoutLeadingZeros))

    plt.tight_layout()
    plt.subplots_adjust(left=0.07, bottom=None, right=None, top=0.95, wspace=0.1, hspace=0.1)
    axes[0, 3].text(x=0.5, y=1.15, s="Number of Recommendations", ha='center', va='center',
                    transform=axes[0, 3].transAxes, fontsize=14)
    plt.savefig("result_plots/results_nr_recs_per_shot__heatmap_grouped.pdf")

    df_list = pd.concat(df_list, axis=1, join='outer')
    # add ci columns to csv:
    offset = 1
    for index, (col, values) in enumerate(ci_dict.items()):
        df_list.insert(index + offset, col, values, True)
        if index % 2 != 0: offset += 1
    df_list.to_csv("result_plots/results_nr_recs_per_shot__heatmap_grouped.csv")
    return


def plotNrRecsOnly10HeatmapGrouped(df_extraction, model_group_list, shot_list, values_in_percentages=False):
    plt.close()
    fig, axes = plt.subplots(
        nrows=len(model_group_list) + 1,
        figsize=(8, 12),
        gridspec_kw={"height_ratios": [len(model_group) for model_group in model_group_list] + [0.3]})
    cm = sns.light_palette("gold", as_cmap=True)

    df_list = list()
    ci_dict = defaultdict(list)
    for i, model_group in enumerate(model_group_list):
        data = {
            "Model": model_group.copy(),
        }
        all_annotations = list()
        for shot, shot_label in shot_list:
            df_extraction_shot = df_extraction.loc[df_extraction["shot"] == shot].reset_index(drop=True)
            shot_values_list = list()
            annotations_list = list()
            for model in model_group:
                df_extraction_shot_model = df_extraction_shot.loc[df_extraction_shot["model"] == model].reset_index(
                    drop=True)
                nr_recs_only_10_percentage = len(
                    df_extraction_shot_model.loc[df_extraction_shot_model['nr_recs'] == 10]) / len(
                    df_extraction_shot_model)
                nr_recs_only_10_list = (
                        [1] * len(df_extraction_shot_model.loc[df_extraction_shot_model['nr_recs'] == 10]) +
                        [0] * len(df_extraction_shot_model.loc[df_extraction_shot_model['nr_recs'] != 10]))
                assert nr_recs_only_10_percentage == np.mean(nr_recs_only_10_list)
                # Bootstrapped CI:
                lower, upper = bootstrapCIV6(list(nr_recs_only_10_list), nr_samples=NR_BOOTSTRAP_SAMPLES)
                ci_dict[shot_label + "_ci_lo"].append(lower)
                ci_dict[shot_label + "_ci_hi"].append(upper)
                shot_values_list.append(nr_recs_only_10_percentage)
                annotations_list.append(("%.4f\n" % nr_recs_only_10_percentage).replace("0.", ".") + (
                        CI_FORMAT_STRING % (lower, upper)).replace("0.", ".\!").replace(", .",
                                                                                        ",\; ."))  # "\;" a thick space; "\!" a negative thin space

            data[shot_label] = shot_values_list
            all_annotations.append(annotations_list)

        # correct model labels:
        for index in range(len(data["Model"])):
            data["Model"][index] = correctModelLabel(data["Model"][index])
        if i == 0:
            ylabel = "Tiny"
        elif i == 1:
            ylabel = "Small"
        elif i == 2:
            ylabel = "Medium"
        elif i == 3:
            ylabel = "Large"
        data[ylabel] = data["Model"]
        del data["Model"]
        df = pd.DataFrame(data)
        df.set_index(ylabel, inplace=True)
        df_list.append(df)
        all_annotations = np.array(all_annotations)
        all_annotations = all_annotations.T
        if values_in_percentages:
            ax = sns.heatmap(df, annot=True, fmt='.2%', cmap=cm, ax=axes[i], cbar=False,
                             annot_kws={"va": 'center_baseline'}, vmin=0, vmax=1)
        else:
            ax = sns.heatmap(df, annot=all_annotations, fmt="", cmap=cm, ax=axes[i], cbar=False,
                             annot_kws={"va": 'center_baseline'}, vmin=0, vmax=1)
        if i == 0:
            axes[i].tick_params(left=True, bottom=False, top=True, right=False, labelleft=True, labelbottom=False,
                                labeltop=True)
            axes[i].xaxis.tick_top()
            axes[i].xaxis.set_label_position('top')
        else:
            axes[i].tick_params(left=True, bottom=False, top=False, right=False, labelleft=True, labelbottom=False)
    if values_in_percentages:
        fig.colorbar(axes[i].get_children()[0], cax=axes[i + 1], orientation="horizontal",
                     format=mtick.PercentFormatter(1.0))
    else:
        fig.colorbar(axes[i].get_children()[0], cax=axes[i + 1], orientation="horizontal",
                     format=mtick.FuncFormatter(formatterWithoutLeadingZeros))

    plt.tight_layout()
    plt.savefig("result_plots/results_nr_recs_per_shot__only_10__heatmap_grouped.pdf")

    df_list = pd.concat(df_list)
    # add ci columns to csv:
    offset = 1
    for index, (col, values) in enumerate(ci_dict.items()):
        df_list.insert(index + offset, col, values, True)
        if index % 2 != 0: offset += 1
    df_list.to_csv("result_plots/results_nr_recs_per_shot__only_10__heatmap_grouped.csv")
    return


def plotNrRecsOnly10HeatmapGroupedByShots(df_extraction, model_group_list, shot_group_list,
                                          values_in_percentages=False):
    plt.close()
    fig, axes = plt.subplots(
        nrows=len(model_group_list) + 1,
        figsize=(6, 14),
        gridspec_kw={"height_ratios": [len(model_group) for model_group in model_group_list] + [0.3]})
    cm = sns.light_palette("gold", as_cmap=True)

    df_list = list()
    ci_dict = defaultdict(list)
    for i, model_group in enumerate(model_group_list):
        data = {
            "Model": model_group.copy(),
        }
        all_annotations = list()
        for shot_label, shot_list in shot_group_list:
            df_extraction_shot = df_extraction.loc[df_extraction["shot"].isin(shot_list)].reset_index(drop=True)
            shot_values_list = list()
            annotations_list = list()
            for model in model_group:
                df_extraction_shot_model = df_extraction_shot.loc[df_extraction_shot["model"] == model].reset_index(
                    drop=True)
                nr_recs_only_10_percentage = len(
                    df_extraction_shot_model.loc[df_extraction_shot_model['nr_recs'] == 10]) / len(
                    df_extraction_shot_model)
                nr_recs_only_10_list = (
                        [1] * len(df_extraction_shot_model.loc[df_extraction_shot_model['nr_recs'] == 10]) +
                        [0] * len(df_extraction_shot_model.loc[df_extraction_shot_model['nr_recs'] != 10]))
                assert nr_recs_only_10_percentage == np.mean(nr_recs_only_10_list)
                # Bootstrapped CI:
                lower, upper = bootstrapCIV6(list(nr_recs_only_10_list), nr_samples=NR_BOOTSTRAP_SAMPLES)
                ci_dict[shot_label + "_ci_lo"].append(lower)
                ci_dict[shot_label + "_ci_hi"].append(upper)
                shot_values_list.append(nr_recs_only_10_percentage)
                annotations_list.append(("%.4f\n" % nr_recs_only_10_percentage).replace("0.", ".") + (
                        CI_FORMAT_STRING % (lower, upper)).replace("0.", ".\!").replace(", .",
                                                                                        ",\; ."))  # "\;" a thick space; "\!" a negative thin space

            # model group average row:
            df_extraction_shot_model_group = df_extraction_shot.loc[
                (df_extraction_shot["model"].isin(model_group))].reset_index(drop=True)
            if len(df_extraction_shot_model_group) == 0:
                shot_values_list.append(None)
                annotations_list.append("")
                continue
            nr_recs_only_10_percentage = len(
                df_extraction_shot_model_group.loc[df_extraction_shot_model_group['nr_recs'] == 10]) / len(
                df_extraction_shot_model_group)
            nr_recs_only_10_list = (
                    [1] * len(df_extraction_shot_model_group.loc[df_extraction_shot_model_group['nr_recs'] == 10]) +
                    [0] * len(df_extraction_shot_model_group.loc[df_extraction_shot_model_group['nr_recs'] != 10]))
            assert nr_recs_only_10_percentage == np.mean(nr_recs_only_10_list)
            # Bootstrapped CI:
            lower, upper = bootstrapCIV6(list(nr_recs_only_10_list), nr_samples=NR_BOOTSTRAP_SAMPLES)
            ci_dict[shot_label + "_ci_lo"].append(lower)
            ci_dict[shot_label + "_ci_hi"].append(upper)
            shot_values_list.append(nr_recs_only_10_percentage)
            annotations_list.append(("%.4f\n" % nr_recs_only_10_percentage).replace("0.", ".") + (
                    CI_FORMAT_STRING % (lower, upper)).replace("0.", ".\!").replace(", .",
                                                                                    ",\; ."))  # "\;" a thick space; "\!" a negative thin space

            data[shot_label] = shot_values_list
            all_annotations.append(annotations_list)

        # Average column:
        shot_values_list = list()
        annotations_list = list()
        for model in model_group:
            df_extraction_model = df_extraction.loc[(df_extraction["model"] == model)].reset_index(drop=True)
            nr_recs_only_10_percentage = len(df_extraction_model.loc[df_extraction_model['nr_recs'] == 10]) / len(
                df_extraction_model)
            nr_recs_only_10_list = ([1] * len(df_extraction_model.loc[df_extraction_model['nr_recs'] == 10]) +
                                    [0] * len(df_extraction_model.loc[df_extraction_model['nr_recs'] != 10]))
            assert nr_recs_only_10_percentage == np.mean(nr_recs_only_10_list)
            # Bootstrapped CI:
            lower, upper = bootstrapCIV6(list(nr_recs_only_10_list), nr_samples=NR_BOOTSTRAP_SAMPLES)
            ci_dict["Average_ci_lo"].append(lower)
            ci_dict["Average_ci_hi"].append(upper)
            shot_values_list.append(nr_recs_only_10_percentage)
            annotations_list.append(("%.4f\n" % nr_recs_only_10_percentage).replace("0.", ".") + (
                    CI_FORMAT_STRING % (lower, upper)).replace("0.", ".\!").replace(", .",
                                                                                    ",\; ."))  # "\;" a thick space; "\!" a negative thin space

        # model group average row:
        df_extraction_model_group = df_extraction.loc[(df_extraction["model"].isin(model_group))].reset_index(drop=True)
        if len(df_extraction_model_group) == 0:
            shot_values_list.append(None)
            annotations_list.append("")
            continue
        nr_recs_only_10_percentage = len(
            df_extraction_model_group.loc[df_extraction_model_group['nr_recs'] == 10]) / len(df_extraction_model_group)
        nr_recs_only_10_list = ([1] * len(df_extraction_model_group.loc[df_extraction_model_group['nr_recs'] == 10]) +
                                [0] * len(df_extraction_model_group.loc[df_extraction_model_group['nr_recs'] != 10]))
        assert nr_recs_only_10_percentage == np.mean(nr_recs_only_10_list)
        # Bootstrapped CI:
        lower, upper = bootstrapCIV6(list(nr_recs_only_10_list), nr_samples=NR_BOOTSTRAP_SAMPLES)
        ci_dict["Average_ci_lo"].append(lower)
        ci_dict["Average_ci_hi"].append(upper)
        shot_values_list.append(nr_recs_only_10_percentage)
        annotations_list.append(
            ("%.4f\n" % nr_recs_only_10_percentage).replace("0.", ".") + (CI_FORMAT_STRING % (lower, upper)).replace(
                "0.", ".\!").replace(", .", ",\; ."))  # "\;" a thick space; "\!" a negative thin space

        data["Average"] = shot_values_list
        all_annotations.append(annotations_list)

        # correct model labels:
        for index in range(len(data["Model"])):
            data["Model"][index] = correctModelLabel(data["Model"][index])
        if i == 0:
            ylabel = "Tiny"
        elif i == 1:
            ylabel = "Small"
        elif i == 2:
            ylabel = "Medium"
        elif i == 3:
            ylabel = "Large"
        data[ylabel] = data["Model"]
        del data["Model"]
        data[ylabel].append("Average")
        df = pd.DataFrame(data)
        df.set_index(ylabel, inplace=True)
        df_list.append(df)
        all_annotations = np.array(all_annotations)
        all_annotations = all_annotations.T
        if values_in_percentages:
            ax = sns.heatmap(df, annot=True, fmt='.2%', cmap=cm, ax=axes[i], cbar=False,
                             annot_kws={"va": 'center_baseline'}, vmin=0, vmax=1)
        else:
            ax = sns.heatmap(df, annot=all_annotations, fmt="", cmap=cm, ax=axes[i], cbar=False,
                             annot_kws={"va": 'center_baseline'}, vmin=0, vmax=1)
        if i == 0:
            axes[i].tick_params(left=True, bottom=False, top=True, right=False, labelleft=True, labelbottom=False,
                                labeltop=True)
            axes[i].xaxis.tick_top()
            axes[i].xaxis.set_label_position('top')
        else:
            axes[i].tick_params(left=True, bottom=False, top=False, right=False, labelleft=True, labelbottom=False)
    if values_in_percentages:
        fig.colorbar(axes[i].get_children()[0], cax=axes[i + 1], orientation="horizontal",
                     format=mtick.PercentFormatter(1.0))
    else:
        fig.colorbar(axes[i].get_children()[0], cax=axes[i + 1], orientation="horizontal",
                     format=mtick.FuncFormatter(formatterWithoutLeadingZeros))

    plt.tight_layout()
    plt.savefig("result_plots/results_nr_recs_per_shot__only_10__heatmap_grouped_by_shots.pdf")

    df_list = pd.concat(df_list)
    # add ci columns to csv:
    offset = 1
    for index, (col, values) in enumerate(ci_dict.items()):
        df_list.insert(index + offset, col, values, True)
        if index % 2 != 0: offset += 1
    df_list.to_csv("result_plots/results_nr_recs_per_shot__only_10__heatmap_grouped_by_shots.csv")
    return


def plotNrRecsOnly10HeatmapGroupedByModelsAndShots(df_extraction, model_group_list, shot_group_list,
                                                   values_in_percentages=False):
    plt.close()
    fig, axes = plt.subplots(
        nrows=2,
        figsize=(5, 2.2),
        gridspec_kw={"height_ratios": [2.2] + [0.1]})
    cm = sns.light_palette("gold", as_cmap=True)

    ci_dict = defaultdict(list)
    data = {
        "Model Group": ["Tiny", "Small", "Medium", "Large"]
    }
    all_annotations = list()
    for shot_label, shot_list in shot_group_list:
        df_extraction_shot = df_extraction.loc[df_extraction["shot"].isin(shot_list)].reset_index(drop=True)
        shot_values_list = list()
        annotations_list = list()
        for i, model_group in enumerate(model_group_list):
            # model group average row:
            df_extraction_shot_model_group = df_extraction_shot.loc[
                (df_extraction_shot["model"].isin(model_group))].reset_index(drop=True)
            if len(df_extraction_shot_model_group) == 0:
                shot_values_list.append(None)
                annotations_list.append("")
                continue
            nr_recs_only_10_percentage = len(
                df_extraction_shot_model_group.loc[df_extraction_shot_model_group['nr_recs'] == 10]) / len(
                df_extraction_shot_model_group)
            nr_recs_only_10_list = (
                    [1] * len(df_extraction_shot_model_group.loc[df_extraction_shot_model_group['nr_recs'] == 10]) +
                    [0] * len(df_extraction_shot_model_group.loc[df_extraction_shot_model_group['nr_recs'] != 10]))
            assert nr_recs_only_10_percentage == np.mean(nr_recs_only_10_list)
            # Bootstrapped CI:
            lower, upper = bootstrapCIV6(list(nr_recs_only_10_list), nr_samples=NR_BOOTSTRAP_SAMPLES)
            ci_dict[shot_label + "_ci_lo"].append(lower)
            ci_dict[shot_label + "_ci_hi"].append(upper)
            shot_values_list.append(nr_recs_only_10_percentage)
            annotations_list.append(("%.4f\n" % nr_recs_only_10_percentage).replace("0.", ".") + (
                    CI_FORMAT_STRING % (lower, upper)).replace("0.", ".\!").replace(", .",
                                                                                    ",\; ."))  # "\;" a thick space; "\!" a negative thin space

        data[shot_label] = shot_values_list
        all_annotations.append(annotations_list)

    # Average column:
    shot_values_list = list()
    annotations_list = list()
    for i, model_group in enumerate(model_group_list):
        # model group average row:
        df_extraction_model_group = df_extraction.loc[(df_extraction["model"].isin(model_group))].reset_index(drop=True)
        if len(df_extraction_model_group) == 0:
            shot_values_list.append(None)
            annotations_list.append("")
            continue
        nr_recs_only_10_percentage = len(
            df_extraction_model_group.loc[df_extraction_model_group['nr_recs'] == 10]) / len(df_extraction_model_group)
        nr_recs_only_10_list = ([1] * len(df_extraction_model_group.loc[df_extraction_model_group['nr_recs'] == 10]) +
                                [0] * len(df_extraction_model_group.loc[df_extraction_model_group['nr_recs'] != 10]))
        assert nr_recs_only_10_percentage == np.mean(nr_recs_only_10_list)
        # Bootstrapped CI:
        lower, upper = bootstrapCIV6(list(nr_recs_only_10_list), nr_samples=NR_BOOTSTRAP_SAMPLES)
        ci_dict["Average_ci_lo"].append(lower)
        ci_dict["Average_ci_hi"].append(upper)
        shot_values_list.append(nr_recs_only_10_percentage)
        annotations_list.append(
            ("%.4f\n" % nr_recs_only_10_percentage).replace("0.", ".") + (CI_FORMAT_STRING % (lower, upper)).replace(
                "0.", ".\!").replace(", .", ",\; ."))  # "\;" a thick space; "\!" a negative thin space

    data["Average"] = shot_values_list
    all_annotations.append(annotations_list)
    df = pd.DataFrame(data)
    df.set_index("Model Group", inplace=True)
    all_annotations = np.array(all_annotations)
    all_annotations = all_annotations.T
    sns.heatmap(df, annot=all_annotations, fmt='', cmap=cm, ax=axes[0], cbar=False, annot_kws={"va": 'center_baseline'},
                vmin=0.24, vmax=1)
    axes[0].set_ylabel("", rotation=0, labelpad=100)
    axes[0].tick_params(left=True, bottom=False, top=True, right=False, labelleft=True, labelbottom=False,
                        labeltop=True)
    axes[0].xaxis.tick_top()
    axes[0].xaxis.set_label_position('top')
    if values_in_percentages:
        fig.colorbar(axes[0].get_children()[0], cax=axes[1], orientation="horizontal",
                     format=mtick.PercentFormatter(1.0))
    else:
        fig.colorbar(axes[0].get_children()[0], cax=axes[1], orientation="horizontal",
                     format=mtick.FuncFormatter(formatterWithoutLeadingZeros))

    plt.tight_layout()
    plt.savefig("result_plots/results_nr_recs_per_shot__only_10__heatmap_grouped_by_models_and_shots.pdf")

    # add ci columns to csv:
    offset = 1
    for index, (col, values) in enumerate(ci_dict.items()):
        df.insert(index + offset, col, values, True)
        if index % 2 != 0: offset += 1
    df.to_csv("result_plots/results_nr_recs_per_shot__only_10__heatmap_grouped_by_models_and_shots.csv")
    return


def plotUniqueMoviesGrouped(df_extraction, model_group_list, shot_list, values_in_percentages=False):
    plt.close()
    fig, axes = plt.subplots(
        nrows=len(model_group_list) + 1,
        figsize=(10, 12),
        gridspec_kw={"height_ratios": [len(model_group) for model_group in model_group_list] + [0.3]})
    cm = sns.light_palette("#a275ac", as_cmap=True)

    df_list = list()
    ci_dict = defaultdict(list)
    for i, model_group in enumerate(model_group_list):
        data = {
            "Model": model_group.copy(),
        }
        all_annotations = list()
        for shot, shot_label in shot_list:
            df_extraction_shot = df_extraction.loc[df_extraction["shot"] == shot].reset_index(drop=True)
            shot_values_list = list()
            annotations_list = list()
            for model in model_group:
                df_extraction_shot_model = df_extraction_shot.loc[
                    (df_extraction_shot["model"] == model) & (df_extraction_shot["nr_recs"] > 0)].reset_index(drop=True)
                unique_percentage = np.mean(
                    df_extraction_shot_model['nr_recs_unique'] / df_extraction_shot_model['nr_recs'])
                unique_list = list(df_extraction_shot_model['nr_recs_unique'] / df_extraction_shot_model['nr_recs'])
                assert unique_percentage == np.mean(unique_list)
                # Bootstrapped CI:
                lower, upper = bootstrapCIV6(list(unique_list), nr_samples=NR_BOOTSTRAP_SAMPLES)
                ci_dict[shot_label + "_ci_lo"].append(lower)
                ci_dict[shot_label + "_ci_hi"].append(upper)
                shot_values_list.append(unique_percentage)
                annotations_list.append(
                    ("%.4f\n" % unique_percentage).replace("0.", ".") + (CI_FORMAT_STRING % (lower, upper)).replace(
                        "0.", ".\!").replace(", .", ",\; ."))  # "\;" a thick space; "\!" a negative thin space
            data[shot_label] = shot_values_list
            all_annotations.append(annotations_list)

        # correct model labels:
        for index in range(len(data["Model"])):
            data["Model"][index] = correctModelLabel(data["Model"][index])
        if i == 0:
            ylabel = "Tiny"
        elif i == 1:
            ylabel = "Small"
        elif i == 2:
            ylabel = "Medium"
        elif i == 3:
            ylabel = "Large"
        data[ylabel] = data["Model"]
        del data["Model"]

        df = pd.DataFrame(data)
        df.set_index(ylabel, inplace=True)
        df_list.append(df)
        all_annotations = np.array(all_annotations)
        all_annotations = all_annotations.T
        if values_in_percentages:
            ax = sns.heatmap(df, annot=True, fmt='.2%', cmap=cm, ax=axes[i], cbar=False,
                             annot_kws={"va": 'center_baseline'}, vmin=0.93, vmax=1)
        else:
            ax = sns.heatmap(df, annot=all_annotations, fmt="", cmap=cm, ax=axes[i], cbar=False,
                             annot_kws={"va": 'center_baseline'}, vmin=0.93, vmax=1)
        if i == 0:
            axes[i].tick_params(left=True, bottom=False, top=True, right=False, labelleft=True, labelbottom=False,
                                labeltop=True)
            axes[i].xaxis.tick_top()
            axes[i].xaxis.set_label_position('top')
        else:
            axes[i].tick_params(left=True, bottom=False, top=False, right=False, labelleft=True, labelbottom=False)
    if values_in_percentages:
        fig.colorbar(axes[i].get_children()[0], cax=axes[i + 1], orientation="horizontal",
                     format=mtick.PercentFormatter(1.0))
    else:
        fig.colorbar(axes[i].get_children()[0], cax=axes[i + 1], orientation="horizontal",
                     format=mtick.FuncFormatter(formatterWithoutLeadingZeros))
    plt.tight_layout()
    plt.subplots_adjust(left=0.18)
    axes[1].text(x=-0.2, y=0.13, s="Unique Movies", ha='center', va='center', transform=axes[1].transAxes, rotation=90,
                 fontsize=14)
    plt.savefig("result_plots/results_unique_movies_grouped.pdf")

    df_list = pd.concat(df_list)
    # add ci columns to csv:
    offset = 1
    for index, (col, values) in enumerate(ci_dict.items()):
        df_list.insert(index + offset, col, values, True)
        if index % 2 != 0: offset += 1
    df_list.to_csv("result_plots/results_unique_movies_grouped.csv")
    return


def plotUniqueMoviesGroupedByShots(df_extraction, model_group_list, shot_group_list, values_in_percentages=False):
    plt.close()
    fig, axes = plt.subplots(
        nrows=len(model_group_list) + 1,
        figsize=(6, 14),
        gridspec_kw={"height_ratios": [len(model_group) for model_group in model_group_list] + [0.3]})
    cm = sns.light_palette("#a275ac", as_cmap=True)

    df_list = list()
    ci_dict = defaultdict(list)

    for i, model_group in enumerate(model_group_list):
        data = {
            "Model": model_group.copy(),
        }
        all_annotations = list()
        for shot_label, shot_list in shot_group_list:
            df_extraction_shot = df_extraction.loc[df_extraction["shot"].isin(shot_list)].reset_index(drop=True)
            shot_values_list = list()
            annotations_list = list()
            for model in model_group:
                df_extraction_shot_model = df_extraction_shot.loc[
                    (df_extraction_shot["model"] == model) & (df_extraction_shot["nr_recs"] > 0)].reset_index(drop=True)
                unique_percentage = np.mean(
                    df_extraction_shot_model['nr_recs_unique'] / df_extraction_shot_model['nr_recs'])
                unique_list = list(df_extraction_shot_model['nr_recs_unique'] / df_extraction_shot_model['nr_recs'])
                assert unique_percentage == np.mean(unique_list)
                # Bootstrapped CI:
                lower, upper = bootstrapCIV6(list(unique_list), nr_samples=NR_BOOTSTRAP_SAMPLES)
                ci_dict[shot_label + "_ci_lo"].append(lower)
                ci_dict[shot_label + "_ci_hi"].append(upper)
                shot_values_list.append(unique_percentage)
                annotations_list.append(
                    ("%.4f\n" % unique_percentage).replace("0.", ".") + (CI_FORMAT_STRING % (lower, upper)).replace(
                        "0.", ".\!").replace(", .", ",\; ."))  # "\;" a thick space; "\!" a negative thin space

            # model group average row:
            df_extraction_shot_model_group = df_extraction_shot.loc[
                (df_extraction_shot["model"].isin(model_group)) & (df_extraction_shot["nr_recs"] > 0)].reset_index(
                drop=True)
            if len(df_extraction_shot_model_group) == 0:
                shot_values_list.append(None)
                annotations_list.append("")
                continue
            unique_percentage = np.mean(
                df_extraction_shot_model_group['nr_recs_unique'] / df_extraction_shot_model_group['nr_recs'])
            unique_list = list(
                df_extraction_shot_model_group['nr_recs_unique'] / df_extraction_shot_model_group['nr_recs'])
            assert unique_percentage == np.mean(unique_list)
            # Bootstrapped CI:
            lower, upper = bootstrapCIV6(list(unique_list), nr_samples=NR_BOOTSTRAP_SAMPLES)
            ci_dict[shot_label + "_ci_lo"].append(lower)
            ci_dict[shot_label + "_ci_hi"].append(upper)
            shot_values_list.append(unique_percentage)
            annotations_list.append(
                ("%.4f\n" % unique_percentage).replace("0.", ".") + (CI_FORMAT_STRING % (lower, upper)).replace("0.",
                                                                                                                ".\!").replace(
                    ", .", ",\; ."))  # "\;" a thick space; "\!" a negative thin space

            data[shot_label] = shot_values_list
            all_annotations.append(annotations_list)

        # Average column:
        shot_values_list = list()
        annotations_list = list()
        for model in model_group:
            df_extraction_model = df_extraction.loc[
                (df_extraction["model"] == model) & (df_extraction["nr_recs"] > 0)].reset_index(drop=True)
            unique_percentage = np.mean(df_extraction_model['nr_recs_unique'] / df_extraction_model['nr_recs'])
            unique_list = list(df_extraction_model['nr_recs_unique'] / df_extraction_model['nr_recs'])
            assert unique_percentage == np.mean(unique_list)
            # Bootstrapped CI:
            lower, upper = bootstrapCIV6(list(unique_list), nr_samples=NR_BOOTSTRAP_SAMPLES)
            ci_dict["Average_ci_lo"].append(lower)
            ci_dict["Average_ci_hi"].append(upper)
            shot_values_list.append(unique_percentage)
            annotations_list.append(
                ("%.4f\n" % unique_percentage).replace("0.", ".") + (CI_FORMAT_STRING % (lower, upper)).replace("0.",
                                                                                                                ".\!").replace(
                    ", .", ",\; ."))  # "\;" a thick space; "\!" a negative thin space

        # model group average row:
        df_extraction_model_group = df_extraction.loc[
            (df_extraction["model"].isin(model_group)) & (df_extraction["nr_recs"] > 0)].reset_index(drop=True)
        if len(df_extraction_model_group) == 0:
            shot_values_list.append(None)
            annotations_list.append("")
            continue
        unique_percentage = np.mean(df_extraction_model_group['nr_recs_unique'] / df_extraction_model_group['nr_recs'])
        unique_list = list(df_extraction_model_group['nr_recs_unique'] / df_extraction_model_group['nr_recs'])
        assert unique_percentage == np.mean(unique_list)
        # Bootstrapped CI:
        lower, upper = bootstrapCIV6(list(unique_list), nr_samples=NR_BOOTSTRAP_SAMPLES)
        ci_dict["Average_ci_lo"].append(lower)
        ci_dict["Average_ci_hi"].append(upper)
        shot_values_list.append(unique_percentage)
        annotations_list.append(
            ("%.4f\n" % unique_percentage).replace("0.", ".") + (CI_FORMAT_STRING % (lower, upper)).replace("0.",
                                                                                                            ".\!").replace(
                ", .", ",\; ."))  # "\;" a thick space; "\!" a negative thin space

        data["Average"] = shot_values_list
        all_annotations.append(annotations_list)

        # correct model labels:
        for index in range(len(data["Model"])):
            data["Model"][index] = correctModelLabel(data["Model"][index])
        if i == 0:
            ylabel = "Tiny"
        elif i == 1:
            ylabel = "Small"
        elif i == 2:
            ylabel = "Medium"
        elif i == 3:
            ylabel = "Large"
        data[ylabel] = data["Model"]
        del data["Model"]
        data[ylabel].append("Average")
        df = pd.DataFrame(data)
        df.set_index(ylabel, inplace=True)
        df_list.append(df)
        all_annotations = np.array(all_annotations)
        all_annotations = all_annotations.T
        if values_in_percentages:
            ax = sns.heatmap(df, annot=True, fmt='.2%', cmap=cm, ax=axes[i], cbar=False,
                             annot_kws={"va": 'center_baseline'}, vmin=0.95, vmax=1)
        else:
            ax = sns.heatmap(df, annot=all_annotations, fmt="", cmap=cm, ax=axes[i], cbar=False,
                             annot_kws={"va": 'center_baseline'}, vmin=0.95, vmax=1)
        if i == 0:
            axes[i].tick_params(left=True, bottom=False, top=True, right=False, labelleft=True, labelbottom=False,
                                labeltop=True)
            axes[i].xaxis.tick_top()
            axes[i].xaxis.set_label_position('top')
        else:
            axes[i].tick_params(left=True, bottom=False, top=False, right=False, labelleft=True, labelbottom=False)
    if values_in_percentages:
        fig.colorbar(axes[i].get_children()[0], cax=axes[i + 1], orientation="horizontal",
                     format=mtick.PercentFormatter(1.0))
    else:
        fig.colorbar(axes[i].get_children()[0], cax=axes[i + 1], orientation="horizontal",
                     format=mtick.FuncFormatter(formatterWithoutLeadingZeros))

    plt.tight_layout()
    plt.savefig("result_plots/results_unique_movies_grouped_by_shots.pdf")

    df_list = pd.concat(df_list)
    # add ci columns to csv:
    offset = 1
    for index, (col, values) in enumerate(ci_dict.items()):
        df_list.insert(index + offset, col, values, True)
        if index % 2 != 0: offset += 1
    df_list.to_csv("result_plots/results_unique_movies_grouped_by_shots.csv")
    return


def plotUniqueMoviesGroupedByModelsAndShots(df_extraction, model_group_list, shot_group_list,
                                            values_in_percentages=False):
    plt.close()
    fig, axes = plt.subplots(
        nrows=2,
        figsize=(5, 2.2),
        gridspec_kw={"height_ratios": [2.2, 0.1]})
    cm = sns.light_palette("#a275ac", as_cmap=True)

    data = {
        "Model Group": ["Tiny", "Small", "Medium", "Large"]
    }
    ci_dict = defaultdict(list)
    all_annotations = list()
    for shot_label, shot_list in shot_group_list:
        df_extraction_shot = df_extraction.loc[df_extraction["shot"].isin(shot_list)].reset_index(drop=True)
        shot_values_list = list()
        annotations_list = list()
        for i, model_group in enumerate(model_group_list):
            # model group average row:
            df_extraction_shot_model_group = df_extraction_shot.loc[
                (df_extraction_shot["model"].isin(model_group)) & (df_extraction_shot["nr_recs"] > 0)].reset_index(
                drop=True)
            if len(df_extraction_shot_model_group) == 0:
                shot_values_list.append(None)
                annotations_list.append("")
                continue
            unique_percentage = np.mean(
                df_extraction_shot_model_group['nr_recs_unique'] / df_extraction_shot_model_group['nr_recs'])
            unique_list = list(
                df_extraction_shot_model_group['nr_recs_unique'] / df_extraction_shot_model_group['nr_recs'])
            assert unique_percentage == np.mean(unique_list)
            # Bootstrapped CI:
            lower, upper = bootstrapCIV6(list(unique_list), nr_samples=NR_BOOTSTRAP_SAMPLES)
            ci_dict[shot_label + "_ci_lo"].append(lower)
            ci_dict[shot_label + "_ci_hi"].append(upper)
            shot_values_list.append(unique_percentage)
            annotations_list.append(
                ("%.4f\n" % unique_percentage).replace("0.", ".") + (CI_FORMAT_STRING % (lower, upper)).replace("0.",
                                                                                                                ".\!").replace(
                    ", .", ",\; ."))  # "\;" a thick space; "\!" a negative thin space
        data[shot_label] = shot_values_list
        all_annotations.append(annotations_list)

    # Average column:
    shot_values_list = list()
    annotations_list = list()
    for i, model_group in enumerate(model_group_list):
        # model group average row:
        df_extraction_model_group = df_extraction.loc[
            (df_extraction["model"].isin(model_group)) & (df_extraction["nr_recs"] > 0)].reset_index(drop=True)
        if len(df_extraction_model_group) == 0:
            shot_values_list.append(None)
            annotations_list.append("")
            continue
        unique_percentage = np.mean(df_extraction_model_group['nr_recs_unique'] / df_extraction_model_group['nr_recs'])
        unique_list = list(df_extraction_model_group['nr_recs_unique'] / df_extraction_model_group['nr_recs'])
        assert unique_percentage == np.mean(unique_list)
        # Bootstrapped CI:
        lower, upper = bootstrapCIV6(list(unique_list), nr_samples=NR_BOOTSTRAP_SAMPLES)
        ci_dict["Average_ci_lo"].append(lower)
        ci_dict["Average_ci_hi"].append(upper)
        shot_values_list.append(unique_percentage)
        annotations_list.append(
            ("%.4f\n" % unique_percentage).replace("0.", ".") + (CI_FORMAT_STRING % (lower, upper)).replace("0.",
                                                                                                            ".\!").replace(
                ", .", ",\; ."))  # "\;" a thick space; "\!" a negative thin space
    data["Average"] = shot_values_list
    all_annotations.append(annotations_list)
    df = pd.DataFrame(data)
    df.set_index("Model Group", inplace=True)
    all_annotations = np.array(all_annotations)
    all_annotations = all_annotations.T
    sns.heatmap(df, annot=all_annotations, fmt='', cmap=cm, ax=axes[0], cbar=False, annot_kws={"va": 'center_baseline'},
                vmin=0.992, vmax=1)
    axes[0].set_ylabel("", rotation=0, labelpad=100)
    axes[0].tick_params(left=True, bottom=False, top=True, right=False, labelleft=True, labelbottom=False,
                        labeltop=True)
    axes[0].xaxis.tick_top()
    axes[0].xaxis.set_label_position('top')
    if values_in_percentages:
        fig.colorbar(axes[0].get_children()[0], cax=axes[1], orientation="horizontal",
                     format=mtick.PercentFormatter(1.0))
    else:
        fig.colorbar(axes[0].get_children()[0], cax=axes[1], orientation="horizontal",
                     format=mtick.FuncFormatter(formatterWithoutLeadingZeros))

    plt.tight_layout()
    plt.savefig("result_plots/results_unique_movies_grouped_by_models_and_shots.pdf")

    # add ci columns to csv:
    offset = 1
    for index, (col, values) in enumerate(ci_dict.items()):
        df.insert(index + offset, col, values, True)
        if index % 2 != 0: offset += 1
    df.to_csv("result_plots/results_unique_movies_grouped_by_models_and_shots.csv")
    return


def plotCorrectYearsGrouped(df_extraction, model_group_list, shot_list, values_in_percentages=False):
    plt.close()
    fig, axes = plt.subplots(
        nrows=len(model_group_list) + 1,
        figsize=(10, 12),
        gridspec_kw={"height_ratios": [len(model_group) for model_group in model_group_list] + [0.3]})
    cm = sns.light_palette("peachpuff", as_cmap=True)

    df_list = list()
    ci_dict = defaultdict(list)

    # to further calculate fraction within correct year span:
    submissions = readSubmissionsFromFile(directory=DATA_DIRECTORY)
    submission2year = dict()

    for i, model_group in enumerate(model_group_list):
        data = {
            "Model": model_group.copy(),
        }
        all_annotations = list()
        for shot, shot_label in shot_list:
            df_extraction_shot = df_extraction.loc[df_extraction["shot"] == shot].reset_index(drop=True)
            shot_values_list = list()
            annotations_list = list()
            for model in model_group:
                df_extraction_shot_model = df_extraction_shot.loc[df_extraction_shot["model"] == model].reset_index(
                    drop=True)
                valid_years_list = list()
                for index, row in df_extraction_shot_model.iterrows():
                    if row["submission_id"] not in submission2year:
                        submission = getSubmissionById(submissions, row["submission_id"])
                        release_year_upper_border = int(
                            str(datetime.datetime.fromtimestamp(float(submission["created_at"])).strftime('%Y')))
                        submission2year[row["submission_id"]] = release_year_upper_border
                    else:
                        release_year_upper_border = submission2year[row["submission_id"]]
                    valid_years_per_submission = list()
                    movie_titles_list = eval(row["rec_list"].lower())
                    for movie_title in movie_titles_list:
                        year = int(movie_title.split("(")[-1].split(")")[0].strip())
                        if year <= release_year_upper_border:
                            valid_years_per_submission.append(1)
                        else:
                            valid_years_per_submission.append(0)
                    if len(valid_years_per_submission) == 0: continue
                    valid_years_list.append(np.mean(valid_years_per_submission))
                # Bootstrapped CI:
                lower, upper = bootstrapCIV6(list(valid_years_list), nr_samples=NR_BOOTSTRAP_SAMPLES)
                ci_dict[shot_label + "_ci_lo"].append(lower)
                ci_dict[shot_label + "_ci_hi"].append(upper)
                shot_values_list.append(np.mean(valid_years_list))
                annotations_list.append(("%.4f\n" % np.mean(valid_years_list)).replace("0.", ".") + (
                        CI_FORMAT_STRING % (lower, upper)).replace("0.", ".\!").replace(", .",
                                                                                        ",\; ."))  # "\;" a thick space; "\!" a negative thin space

            data[shot_label] = shot_values_list
            all_annotations.append(annotations_list)

        # correct model labels:
        for index in range(len(data["Model"])):
            data["Model"][index] = correctModelLabel(data["Model"][index])
        if i == 0:
            ylabel = "Tiny"
        elif i == 1:
            ylabel = "Small"
        elif i == 2:
            ylabel = "Medium"
        elif i == 3:
            ylabel = "Large"
        data[ylabel] = data["Model"]
        del data["Model"]
        df = pd.DataFrame(data)
        df.set_index(ylabel, inplace=True)
        df_list.append(df)
        all_annotations = np.array(all_annotations)
        all_annotations = all_annotations.T
        if values_in_percentages:
            ax = sns.heatmap(df, annot=True, fmt='.2%', cmap=cm, ax=axes[i], cbar=False,
                             annot_kws={"va": 'center_baseline'}, vmin=0.7, vmax=1)
        else:
            ax = sns.heatmap(df, annot=all_annotations, fmt="", cmap=cm, ax=axes[i], cbar=False,
                             annot_kws={"va": 'center_baseline'}, vmin=0.7, vmax=1)
        if i == 0:
            axes[i].tick_params(left=True, bottom=False, top=True, right=False, labelleft=True, labelbottom=False,
                                labeltop=True)
            axes[i].xaxis.tick_top()
            axes[i].xaxis.set_label_position('top')
        else:
            axes[i].tick_params(left=True, bottom=False, top=False, right=False, labelleft=True, labelbottom=False)
    if values_in_percentages:
        fig.colorbar(axes[i].get_children()[0], cax=axes[i + 1], orientation="horizontal",
                     format=mtick.PercentFormatter(1.0))
    else:
        fig.colorbar(axes[i].get_children()[0], cax=axes[i + 1], orientation="horizontal",
                     format=mtick.FuncFormatter(formatterWithoutLeadingZeros))

    plt.tight_layout()
    plt.subplots_adjust(left=0.18)
    axes[1].text(x=-0.2, y=0.13, s="Movies Released Before Request", ha='center', va='center',
                 transform=axes[1].transAxes, rotation=90, fontsize=14)
    plt.savefig("result_plots/results_correct_years_grouped.pdf")

    df_list = pd.concat(df_list)
    # add ci columns to csv:
    offset = 1
    for index, (col, values) in enumerate(ci_dict.items()):
        df_list.insert(index + offset, col, values, True)
        if index % 2 != 0: offset += 1
    df_list.to_csv("result_plots/results_correct_years_grouped.csv")
    return


def plotCorrectYearsGroupedByShots(df_extraction, model_group_list, shot_group_list, values_in_percentages=False):
    plt.close()
    fig, axes = plt.subplots(
        nrows=len(model_group_list) + 1,
        figsize=(6, 14),
        gridspec_kw={"height_ratios": [len(model_group) for model_group in model_group_list] + [0.3]})
    cm = sns.light_palette("peachpuff", as_cmap=True)

    df_list = list()
    ci_dict = defaultdict(list)

    # to further calculate fraction within correct year span:
    submissions = readSubmissionsFromFile(directory=DATA_DIRECTORY)
    submission2year = dict()

    for i, model_group in enumerate(model_group_list):
        data = {
            "Model": model_group.copy(),
        }
        all_annotations = list()
        for shot_label, shot_list in shot_group_list:
            df_extraction_shot = df_extraction.loc[df_extraction["shot"].isin(shot_list)].reset_index(drop=True)
            shot_values_list = list()
            annotations_list = list()
            for model in model_group:
                df_extraction_shot_model = df_extraction_shot.loc[df_extraction_shot["model"] == model].reset_index(
                    drop=True)
                valid_years_list = list()
                for index, row in df_extraction_shot_model.iterrows():
                    if row["submission_id"] not in submission2year:
                        submission = getSubmissionById(submissions, row["submission_id"])
                        release_year_upper_border = int(
                            str(datetime.datetime.fromtimestamp(float(submission["created_at"])).strftime('%Y')))
                        submission2year[row["submission_id"]] = release_year_upper_border
                    else:
                        release_year_upper_border = submission2year[row["submission_id"]]
                    valid_years_per_submission = list()
                    movie_titles_list = eval(row["rec_list"].lower())
                    for movie_title in movie_titles_list:
                        year = int(movie_title.split("(")[-1].split(")")[0].strip())
                        if year <= release_year_upper_border:
                            valid_years_per_submission.append(1)
                        else:
                            valid_years_per_submission.append(0)
                    if len(valid_years_per_submission) == 0: continue
                    valid_years_list.append(np.mean(valid_years_per_submission))
                # Bootstrapped CI:
                lower, upper = bootstrapCIV6(list(valid_years_list), nr_samples=NR_BOOTSTRAP_SAMPLES)
                ci_dict[shot_label + "_ci_lo"].append(lower)
                ci_dict[shot_label + "_ci_hi"].append(upper)
                shot_values_list.append(np.mean(valid_years_list))
                annotations_list.append(("%.4f\n" % np.mean(valid_years_list)).replace("0.", ".") + (
                        CI_FORMAT_STRING % (lower, upper)).replace("0.", ".\!").replace(", .",
                                                                                        ",\; ."))  # "\;" a thick space; "\!" a negative thin space

            # model group average row:
            df_extraction_shot_model_group = df_extraction_shot.loc[
                df_extraction_shot["model"].isin(model_group)].reset_index(drop=True)
            if len(df_extraction_shot_model_group) == 0:
                shot_values_list.append(None)
                annotations_list.append("")
                continue
            valid_years_list = list()
            for index, row in df_extraction_shot_model_group.iterrows():
                if row["submission_id"] not in submission2year:
                    submission = getSubmissionById(submissions, row["submission_id"])
                    release_year_upper_border = int(
                        str(datetime.datetime.fromtimestamp(float(submission["created_at"])).strftime('%Y')))
                    submission2year[row["submission_id"]] = release_year_upper_border
                else:
                    release_year_upper_border = submission2year[row["submission_id"]]
                valid_years_per_submission = list()
                movie_titles_list = eval(row["rec_list"].lower())
                for movie_title in movie_titles_list:
                    year = int(movie_title.split("(")[-1].split(")")[0].strip())
                    if year <= release_year_upper_border:
                        valid_years_per_submission.append(1)
                    else:
                        valid_years_per_submission.append(0)
                if len(valid_years_per_submission) == 0: continue
                valid_years_list.append(np.mean(valid_years_per_submission))
            # Bootstrapped CI:
            lower, upper = bootstrapCIV6(list(valid_years_list), nr_samples=NR_BOOTSTRAP_SAMPLES)
            ci_dict[shot_label + "_ci_lo"].append(lower)
            ci_dict[shot_label + "_ci_hi"].append(upper)
            shot_values_list.append(np.mean(valid_years_list))
            annotations_list.append(
                ("%.4f\n" % np.mean(valid_years_list)).replace("0.", ".") + (CI_FORMAT_STRING % (lower, upper)).replace(
                    "0.", ".\!").replace(", .", ",\; ."))  # "\;" a thick space; "\!" a negative thin space

            data[shot_label] = shot_values_list
            all_annotations.append(annotations_list)

        # Average column:
        shot_values_list = list()
        annotations_list = list()
        for model in model_group:
            df_extraction_model = df_extraction.loc[df_extraction["model"] == model].reset_index(drop=True)
            valid_years_list = list()
            for index, row in df_extraction_model.iterrows():
                if row["submission_id"] not in submission2year:
                    submission = getSubmissionById(submissions, row["submission_id"])
                    release_year_upper_border = int(
                        str(datetime.datetime.fromtimestamp(float(submission["created_at"])).strftime('%Y')))
                    submission2year[row["submission_id"]] = release_year_upper_border
                else:
                    release_year_upper_border = submission2year[row["submission_id"]]
                valid_years_per_submission = list()
                movie_titles_list = eval(row["rec_list"].lower())
                for movie_title in movie_titles_list:
                    year = int(movie_title.split("(")[-1].split(")")[0].strip())
                    if year <= release_year_upper_border:
                        valid_years_per_submission.append(1)
                    else:
                        valid_years_per_submission.append(0)
                if len(valid_years_per_submission) == 0: continue
                valid_years_list.append(np.mean(valid_years_per_submission))

            # Bootstrapped CI:
            lower, upper = bootstrapCIV6(list(valid_years_list), nr_samples=NR_BOOTSTRAP_SAMPLES)
            ci_dict["Average_ci_lo"].append(lower)
            ci_dict["Average_ci_hi"].append(upper)
            shot_values_list.append(np.mean(valid_years_list))
            annotations_list.append(
                ("%.4f\n" % np.mean(valid_years_list)).replace("0.", ".") + (CI_FORMAT_STRING % (lower, upper)).replace(
                    "0.", ".\!").replace(", .", ",\; ."))  # "\;" a thick space; "\!" a negative thin space

        # model group average row:
        df_extraction_model_group = df_extraction.loc[df_extraction["model"].isin(model_group)].reset_index(drop=True)
        if len(df_extraction_model_group) == 0:
            shot_values_list.append(None)
            annotations_list.append("")
            continue
        valid_years_list = list()
        for index, row in df_extraction_model_group.iterrows():
            if row["submission_id"] not in submission2year:
                submission = getSubmissionById(submissions, row["submission_id"])
                release_year_upper_border = int(
                    str(datetime.datetime.fromtimestamp(float(submission["created_at"])).strftime('%Y')))
                submission2year[row["submission_id"]] = release_year_upper_border
            else:
                release_year_upper_border = submission2year[row["submission_id"]]
            valid_years_per_submission = list()
            movie_titles_list = eval(row["rec_list"].lower())
            for movie_title in movie_titles_list:
                year = int(movie_title.split("(")[-1].split(")")[0].strip())
                if year <= release_year_upper_border:
                    valid_years_per_submission.append(1)
                else:
                    valid_years_per_submission.append(0)
            if len(valid_years_per_submission) == 0: continue
            valid_years_list.append(np.mean(valid_years_per_submission))
        # Bootstrapped CI:
        lower, upper = bootstrapCIV6(list(valid_years_list), nr_samples=NR_BOOTSTRAP_SAMPLES)
        ci_dict["Average_ci_lo"].append(lower)
        ci_dict["Average_ci_hi"].append(upper)
        shot_values_list.append(np.mean(valid_years_list))
        annotations_list.append(
            ("%.4f\n" % np.mean(valid_years_list)).replace("0.", ".") + (CI_FORMAT_STRING % (lower, upper)).replace(
                "0.", ".\!").replace(", .", ",\; ."))  # "\;" a thick space; "\!" a negative thin space

        data["Average"] = shot_values_list
        all_annotations.append(annotations_list)

        # correct model labels:
        for index in range(len(data["Model"])):
            data["Model"][index] = correctModelLabel(data["Model"][index])
        if i == 0:
            ylabel = "Tiny"
        elif i == 1:
            ylabel = "Small"
        elif i == 2:
            ylabel = "Medium"
        elif i == 3:
            ylabel = "Large"
        data[ylabel] = data["Model"]
        del data["Model"]
        data[ylabel].append("Average")
        df = pd.DataFrame(data)
        df.set_index(ylabel, inplace=True)
        df_list.append(df)
        all_annotations = np.array(all_annotations)
        all_annotations = all_annotations.T
        if values_in_percentages:
            ax = sns.heatmap(df, annot=True, fmt='.2%', cmap=cm, ax=axes[i], cbar=False,
                             annot_kws={"va": 'center_baseline'}, vmin=0.7, vmax=1)
        else:
            ax = sns.heatmap(df, annot=all_annotations, fmt="", cmap=cm, ax=axes[i], cbar=False,
                             annot_kws={"va": 'center_baseline'}, vmin=0.7, vmax=1)
        if i == 0:
            axes[i].tick_params(left=True, bottom=False, top=True, right=False, labelleft=True, labelbottom=False,
                                labeltop=True)
            axes[i].xaxis.tick_top()
            axes[i].xaxis.set_label_position('top')
        else:
            axes[i].tick_params(left=True, bottom=False, top=False, right=False, labelleft=True, labelbottom=False)
    if values_in_percentages:
        fig.colorbar(axes[i].get_children()[0], cax=axes[i + 1], orientation="horizontal",
                     format=mtick.PercentFormatter(1.0))
    else:
        fig.colorbar(axes[i].get_children()[0], cax=axes[i + 1], orientation="horizontal",
                     format=mtick.FuncFormatter(formatterWithoutLeadingZeros))

    plt.tight_layout()
    plt.savefig("result_plots/results_correct_years_grouped_by_shots.pdf")

    df_list = pd.concat(df_list)
    # add ci columns to csv:
    offset = 1
    for index, (col, values) in enumerate(ci_dict.items()):
        df_list.insert(index + offset, col, values, True)
        if index % 2 != 0: offset += 1
    df_list.to_csv("result_plots/results_correct_years_grouped_by_shots.csv")
    return


def plotCorrectYearsGroupedByModelsAndShots(df_extraction, model_group_list, shot_group_list,
                                            values_in_percentages=False):
    plt.close()
    fig, axes = plt.subplots(
        nrows=2,
        figsize=(5, 2.2),
        gridspec_kw={"height_ratios": [2.2, 0.1]})
    cm = sns.light_palette("peachpuff", as_cmap=True)

    ci_dict = defaultdict(list)
    data = {
        "Model Group": ["Tiny", "Small", "Medium", "Large"]
    }

    # to further calculate fraction within correct year span:
    submissions = readSubmissionsFromFile(directory=DATA_DIRECTORY)
    submission2year = dict()

    all_annotations = list()
    for shot_label, shot_list in shot_group_list:
        df_extraction_shot = df_extraction.loc[df_extraction["shot"].isin(shot_list)].reset_index(drop=True)
        shot_values_list = list()
        annotations_list = list()
        for i, model_group in enumerate(model_group_list):
            # model group average row:
            df_extraction_shot_model_group = df_extraction_shot.loc[
                df_extraction_shot["model"].isin(model_group)].reset_index(drop=True)
            if len(df_extraction_shot_model_group) == 0:
                shot_values_list.append(None)
                annotations_list.append("")
                continue
            valid_years_list = list()
            for index, row in df_extraction_shot_model_group.iterrows():
                if row["submission_id"] not in submission2year:
                    submission = getSubmissionById(submissions, row["submission_id"])
                    release_year_upper_border = int(
                        str(datetime.datetime.fromtimestamp(float(submission["created_at"])).strftime('%Y')))
                    submission2year[row["submission_id"]] = release_year_upper_border
                else:
                    release_year_upper_border = submission2year[row["submission_id"]]
                valid_years_per_submission = list()
                movie_titles_list = eval(row["rec_list"].lower())
                for movie_title in movie_titles_list:
                    year = int(movie_title.split("(")[-1].split(")")[0].strip())
                    if year <= release_year_upper_border:
                        valid_years_per_submission.append(1)
                    else:
                        valid_years_per_submission.append(0)
                if len(valid_years_per_submission) == 0: continue
                valid_years_list.append(np.mean(valid_years_per_submission))
            # Bootstrapped CI:
            lower, upper = bootstrapCIV6(list(valid_years_list), nr_samples=NR_BOOTSTRAP_SAMPLES)
            ci_dict[shot_label + "_ci_lo"].append(lower)
            ci_dict[shot_label + "_ci_hi"].append(upper)
            shot_values_list.append(np.mean(valid_years_list))
            annotations_list.append(
                ("%.4f\n" % np.mean(valid_years_list)).replace("0.", ".") + (CI_FORMAT_STRING % (lower, upper)).replace(
                    "0.", ".\!").replace(", .", ",\; ."))  # "\;" a thick space; "\!" a negative thin space

        data[shot_label] = shot_values_list
        all_annotations.append(annotations_list)

    # Average column:
    shot_values_list = list()
    annotations_list = list()
    for i, model_group in enumerate(model_group_list):
        # model group average row:
        df_extraction_model_group = df_extraction.loc[df_extraction["model"].isin(model_group)].reset_index(drop=True)
        if len(df_extraction_model_group) == 0:
            shot_values_list.append(None)
            annotations_list.append("")
            continue
        valid_years_list = list()
        for index, row in df_extraction_model_group.iterrows():
            if row["submission_id"] not in submission2year:
                submission = getSubmissionById(submissions, row["submission_id"])
                release_year_upper_border = int(
                    str(datetime.datetime.fromtimestamp(float(submission["created_at"])).strftime('%Y')))
                submission2year[row["submission_id"]] = release_year_upper_border
            else:
                release_year_upper_border = submission2year[row["submission_id"]]
            valid_years_per_submission = list()
            movie_titles_list = eval(row["rec_list"].lower())
            for movie_title in movie_titles_list:
                year = int(movie_title.split("(")[-1].split(")")[0].strip())
                if year <= release_year_upper_border:
                    valid_years_per_submission.append(1)
                else:
                    valid_years_per_submission.append(0)
            if len(valid_years_per_submission) == 0: continue
            valid_years_list.append(np.mean(valid_years_per_submission))
        # Bootstrapped CI:
        lower, upper = bootstrapCIV6(list(valid_years_list), nr_samples=NR_BOOTSTRAP_SAMPLES)
        ci_dict["Average_ci_lo"].append(lower)
        ci_dict["Average_ci_hi"].append(upper)
        shot_values_list.append(np.mean(valid_years_list))
        annotations_list.append(
            ("%.4f\n" % np.mean(valid_years_list)).replace("0.", ".") + (CI_FORMAT_STRING % (lower, upper)).replace(
                "0.", ".\!").replace(", .", ",\; ."))  # "\;" a thick space; "\!" a negative thin space

    data["Average"] = shot_values_list
    all_annotations.append(annotations_list)
    df = pd.DataFrame(data)
    df.set_index("Model Group", inplace=True)
    all_annotations = np.array(all_annotations)
    all_annotations = all_annotations.T
    sns.heatmap(df, annot=all_annotations, fmt='', cmap=cm, ax=axes[0], cbar=False, annot_kws={"va": 'center_baseline'},
                vmin=0.95, vmax=1)
    axes[0].set_ylabel("", rotation=0, labelpad=100)
    axes[0].tick_params(left=True, bottom=False, top=True, right=False, labelleft=True, labelbottom=False,
                        labeltop=True)
    axes[0].xaxis.tick_top()
    axes[0].xaxis.set_label_position('top')
    if values_in_percentages:
        fig.colorbar(axes[0].get_children()[0], cax=axes[1], orientation="horizontal",
                     format=mtick.PercentFormatter(1.0))
    else:
        fig.colorbar(axes[0].get_children()[0], cax=axes[1], orientation="horizontal",
                     format=mtick.FuncFormatter(formatterWithoutLeadingZeros))

    plt.tight_layout()
    plt.savefig("result_plots/results_correct_years_grouped_by_models_and_shots.pdf")

    # add ci columns to csv:
    offset = 1
    for index, (col, values) in enumerate(ci_dict.items()):
        df.insert(index + offset, col, values, True)
        if index % 2 != 0: offset += 1
    df.to_csv("result_plots/results_correct_years_grouped_by_models_and_shots.csv")
    return


def plotDiversityOverCyclesJaccardGrouped(model_group_list, shot_list, values_in_percentages=False):
    df_extraction = pd.read_csv(os.path.join(DIRECTORY, EXTRACTION_RESULTS_FILENAME))

    plt.close()
    fig, axes = plt.subplots(
        nrows=len(model_group_list) + 1,
        figsize=(10, 12),
        gridspec_kw={"height_ratios": [len(model_group) for model_group in model_group_list] + [0.3]})
    cm = sns.light_palette((20, 60, 50), input="husl", as_cmap=True)

    df_list = list()
    ci_dict = defaultdict(list)
    for i, model_group in enumerate(model_group_list):
        data = {
            "Model": model_group.copy(),
        }
        all_annotations = list()
        for shot, shot_label in shot_list:
            df_extraction_shot = df_extraction.loc[df_extraction["shot"] == shot].reset_index(drop=True)
            all_recommendations_jaccard_diversity_per_submission_over_all_cycles = list()
            shot_values_list = list()
            annotations_list = list()
            for model in model_group:
                df_extraction_shot_model = df_extraction_shot.loc[df_extraction_shot["model"] == model].reset_index(
                    drop=True)
                recommendations_jaccard_diversity_per_submission_over_all_cycles = list()
                for submission_id in sorted(df_extraction_shot_model["submission_id"].unique()):
                    df_extraction_shot_model_submission_id = df_extraction_shot_model.loc[
                        df_extraction_shot_model["submission_id"] == submission_id].reset_index(drop=True)
                    cycle_dict = dict()
                    for index, row in df_extraction_shot_model_submission_id.iterrows():
                        movie_titles_list_without_duplicates_upper_limit = eval(
                            row["rec_list_unique_upper_limit"].lower())
                        assert (type(movie_titles_list_without_duplicates_upper_limit) is list)
                        assert (len(movie_titles_list_without_duplicates_upper_limit) == row[
                            "nr_recs_unique_upper_limit"])
                        cycle_dict[row["cycle"]] = movie_titles_list_without_duplicates_upper_limit
                    cycle_pool = set(cycle_dict)
                    cycle_pool_copy = cycle_pool.copy()
                    counter = 0
                    jaccard_list = list()
                    for cycle1 in cycle_pool_copy:
                        cycle_pool.remove(cycle1)
                        for cycle2 in cycle_pool:
                            counter += 1
                            if len(cycle_dict[cycle1]) == 0 or len(cycle_dict[cycle2]) == 0: continue
                            jaccard = len(set(cycle_dict[cycle1]) & set(cycle_dict[cycle2])) / len(
                                set(cycle_dict[cycle1]) | set(cycle_dict[cycle2]))
                            jaccard_list.append(1 - jaccard)
                    if len(jaccard_list) == 0: continue
                    recommendations_jaccard_diversity_per_submission_over_all_cycles.append(np.mean(jaccard_list))
                    all_recommendations_jaccard_diversity_per_submission_over_all_cycles.append(np.mean(jaccard_list))
                if len(recommendations_jaccard_diversity_per_submission_over_all_cycles) == 0:
                    recommendations_jaccard_diversity_per_submission_over_all_cycles.append(1.0)
                # Bootstrapped CI:
                lower, upper = bootstrapCIV6(list(recommendations_jaccard_diversity_per_submission_over_all_cycles),
                                             nr_samples=NR_BOOTSTRAP_SAMPLES)
                ci_dict[shot_label + "_ci_lo"].append(lower)
                ci_dict[shot_label + "_ci_hi"].append(upper)
                shot_values_list.append(np.mean(recommendations_jaccard_diversity_per_submission_over_all_cycles))
                annotations_list.append(
                    ("%.4f\n" % np.mean(recommendations_jaccard_diversity_per_submission_over_all_cycles)).replace("0.",
                                                                                                                   ".") + (
                            CI_FORMAT_STRING % (lower, upper)).replace("0.", ".\!").replace(", .",
                                                                                            ",\; ."))  # "\;" a thick space; "\!" a negative thin space

            data[shot_label] = shot_values_list
            all_annotations.append(annotations_list)

        # correct model labels:
        for index in range(len(data["Model"])):
            data["Model"][index] = correctModelLabel(data["Model"][index])
        if i == 0:
            ylabel = "Tiny"
        elif i == 1:
            ylabel = "Small"
        elif i == 2:
            ylabel = "Medium"
        elif i == 3:
            ylabel = "Large"
        data[ylabel] = data["Model"]
        del data["Model"]
        df = pd.DataFrame(data)
        df.set_index(ylabel, inplace=True)
        df_list.append(df)
        all_annotations = np.array(all_annotations)
        all_annotations = all_annotations.T
        if values_in_percentages:
            ax = sns.heatmap(df, annot=True, fmt='.2%', cmap=cm, ax=axes[i], cbar=False,
                             annot_kws={"va": 'center_baseline'}, vmin=0.45, vmax=1)
        else:
            ax = sns.heatmap(df, annot=all_annotations, fmt="", cmap=cm, ax=axes[i], cbar=False,
                             annot_kws={"va": 'center_baseline'}, vmin=0.45, vmax=1)
        if i == 0:
            axes[i].tick_params(left=True, bottom=False, top=True, right=False, labelleft=True, labelbottom=False,
                                labeltop=True)
            axes[i].xaxis.tick_top()
            axes[i].xaxis.set_label_position('top')
        else:
            axes[i].tick_params(left=True, bottom=False, top=False, right=False, labelleft=True, labelbottom=False)
    if values_in_percentages:
        fig.colorbar(axes[i].get_children()[0], cax=axes[i + 1], orientation="horizontal",
                     format=mtick.PercentFormatter(1.0))
    else:
        fig.colorbar(axes[i].get_children()[0], cax=axes[i + 1], orientation="horizontal",
                     format=mtick.FuncFormatter(formatterWithoutLeadingZeros))

    plt.tight_layout()
    plt.subplots_adjust(left=0.18)
    axes[1].text(x=-0.2, y=0.13, s="Diversity@10", ha='center', va='center', transform=axes[1].transAxes, rotation=90,
                 fontsize=14)
    plt.savefig("result_plots/results_diversity_over_cycles_jaccard_grouped.pdf")

    df_list = pd.concat(df_list)
    # add ci columns to csv:
    offset = 1
    for index, (col, values) in enumerate(ci_dict.items()):
        df_list.insert(index + offset, col, values, True)
        if index % 2 != 0: offset += 1
    df_list.to_csv("result_plots/results_diversity_over_cycles_jaccard_grouped.csv")
    return


def plotDiversityOverCyclesJaccardGroupedByShots(model_group_list, shot_group_list, values_in_percentages=False):
    df_extraction = pd.read_csv(os.path.join(DIRECTORY, EXTRACTION_RESULTS_FILENAME))

    plt.close()
    fig, axes = plt.subplots(
        nrows=len(model_group_list) + 1,
        figsize=(6, 14),
        gridspec_kw={"height_ratios": [len(model_group) for model_group in model_group_list] + [0.3]})
    cm = sns.light_palette((20, 60, 50), input="husl", as_cmap=True)

    df_list = list()
    ci_dict = defaultdict(list)
    all_shot_values_list_for_overall_row = defaultdict(list)
    for i, model_group in enumerate(model_group_list):
        data = {
            "Model": model_group.copy(),
        }
        all_annotations = list()
        for shot_label, shot_list in shot_group_list:
            df_extraction_shot = df_extraction.loc[df_extraction["shot"].isin(shot_list)].reset_index(drop=True)
            all_recommendations_jaccard_diversity_per_submission_over_all_cycles = list()
            shot_values_list = list()
            annotations_list = list()
            for model in model_group:
                df_extraction_shot_model = df_extraction_shot.loc[df_extraction_shot["model"] == model].reset_index(
                    drop=True)
                recommendations_jaccard_diversity_per_submission_over_all_cycles = list()
                # calc jaccard only between same shots in shot_list:
                for shot in shot_list:
                    df_extraction_shot_model_shot = df_extraction_shot_model.loc[
                        df_extraction_shot_model["shot"] == shot].reset_index(drop=True)
                    for submission_id in sorted(df_extraction_shot_model_shot["submission_id"].unique()):
                        df_extraction_shot_model_shot_submission_id = df_extraction_shot_model_shot.loc[
                            df_extraction_shot_model_shot["submission_id"] == submission_id].reset_index(drop=True)
                        cycle_dict = dict()
                        for index, row in df_extraction_shot_model_shot_submission_id.iterrows():
                            movie_titles_list_without_duplicates_upper_limit = eval(
                                row["rec_list_unique_upper_limit"].lower())
                            assert (type(movie_titles_list_without_duplicates_upper_limit) is list)
                            assert (len(movie_titles_list_without_duplicates_upper_limit) == row[
                                "nr_recs_unique_upper_limit"])
                            cycle_dict[row["cycle"]] = movie_titles_list_without_duplicates_upper_limit
                        cycle_pool = set(cycle_dict)
                        cycle_pool_copy = cycle_pool.copy()
                        counter = 0
                        jaccard_list = list()
                        for cycle1 in cycle_pool_copy:
                            cycle_pool.remove(cycle1)
                            for cycle2 in cycle_pool:
                                counter += 1
                                if len(cycle_dict[cycle1]) == 0 or len(cycle_dict[cycle2]) == 0: continue
                                jaccard = len(set(cycle_dict[cycle1]) & set(cycle_dict[cycle2])) / len(
                                    set(cycle_dict[cycle1]) | set(cycle_dict[cycle2]))
                                jaccard_list.append(1 - jaccard)
                        if len(jaccard_list) == 0: continue
                        recommendations_jaccard_diversity_per_submission_over_all_cycles.append(np.mean(jaccard_list))
                        all_recommendations_jaccard_diversity_per_submission_over_all_cycles.append(
                            np.mean(jaccard_list))
                if len(recommendations_jaccard_diversity_per_submission_over_all_cycles) == 0:
                    recommendations_jaccard_diversity_per_submission_over_all_cycles.append(1.0)
                # Bootstrapped CI:
                lower, upper = bootstrapCIV6(list(recommendations_jaccard_diversity_per_submission_over_all_cycles),
                                             nr_samples=NR_BOOTSTRAP_SAMPLES)
                ci_dict[shot_label + "_ci_lo"].append(lower)
                ci_dict[shot_label + "_ci_hi"].append(upper)
                shot_values_list.append(np.mean(recommendations_jaccard_diversity_per_submission_over_all_cycles))
                annotations_list.append(
                    ("%.4f\n" % np.mean(recommendations_jaccard_diversity_per_submission_over_all_cycles)).replace("0.",
                                                                                                                   ".") + (
                            CI_FORMAT_STRING % (lower, upper)).replace("0.", ".\!").replace(", .",
                                                                                            ",\; ."))  # "\;" a thick space; "\!" a negative thin space

            # Bootstrapped CI:
            lower, upper = bootstrapCIV6(list(all_recommendations_jaccard_diversity_per_submission_over_all_cycles),
                                         nr_samples=NR_BOOTSTRAP_SAMPLES)
            ci_dict[shot_label + "_ci_lo"].append(lower)
            ci_dict[shot_label + "_ci_hi"].append(upper)
            all_shot_values_list_for_overall_row[
                shot_label] += all_recommendations_jaccard_diversity_per_submission_over_all_cycles
            shot_values_list.append(np.mean(all_recommendations_jaccard_diversity_per_submission_over_all_cycles))
            annotations_list.append(
                ("%.4f\n" % np.mean(all_recommendations_jaccard_diversity_per_submission_over_all_cycles)).replace("0.",
                                                                                                                   ".") + (
                        CI_FORMAT_STRING % (lower, upper)).replace("0.", ".\!").replace(", .",
                                                                                        ",\; ."))  # "\;" a thick space; "\!" a negative thin space

            data[shot_label] = shot_values_list
            all_annotations.append(annotations_list)

        # Average column:
        all_recommendations_jaccard_diversity_per_submission_over_all_cycles = list()
        shot_values_list = list()
        annotations_list = list()
        for model in model_group:
            df_extraction_model = df_extraction.loc[df_extraction["model"] == model].reset_index(drop=True)
            recommendations_jaccard_diversity_per_submission_over_all_cycles = list()
            for submission_id in sorted(df_extraction_model["submission_id"].unique()):
                df_extraction_model_submission_id = df_extraction_model.loc[
                    df_extraction_model["submission_id"] == submission_id].reset_index(drop=True)
                cycle_dict = dict()
                for index, row in df_extraction_model_submission_id.iterrows():
                    movie_titles_list_without_duplicates_upper_limit = eval(row["rec_list_unique_upper_limit"].lower())
                    assert (type(movie_titles_list_without_duplicates_upper_limit) is list)
                    assert (len(movie_titles_list_without_duplicates_upper_limit) == row["nr_recs_unique_upper_limit"])
                    cycle_dict[row["shot"] + str(row["cycle"])] = movie_titles_list_without_duplicates_upper_limit
                cycle_pool = set(cycle_dict)
                cycle_pool_copy = cycle_pool.copy()
                counter = 0
                jaccard_list = list()
                for cycle1 in cycle_pool_copy:
                    cycle_pool.remove(cycle1)
                    for cycle2 in cycle_pool:
                        counter += 1
                        if len(cycle_dict[cycle1]) == 0 or len(cycle_dict[cycle2]) == 0: continue
                        jaccard = len(set(cycle_dict[cycle1]) & set(cycle_dict[cycle2])) / len(
                            set(cycle_dict[cycle1]) | set(cycle_dict[cycle2]))
                        intersection = len(set(cycle_dict[cycle1]).intersection(set(cycle_dict[cycle2])))
                        union = len(set(cycle_dict[cycle1]).union(set(cycle_dict[cycle2])))
                        assert (intersection / union) == jaccard
                        jaccard_list.append(1 - jaccard)
                if len(jaccard_list) == 0: continue
                recommendations_jaccard_diversity_per_submission_over_all_cycles.append(np.mean(jaccard_list))
                all_recommendations_jaccard_diversity_per_submission_over_all_cycles.append(np.mean(jaccard_list))

            if len(recommendations_jaccard_diversity_per_submission_over_all_cycles) == 0:
                recommendations_jaccard_diversity_per_submission_over_all_cycles.append(1.0)
            # Bootstrapped CI:
            lower, upper = bootstrapCIV6(list(recommendations_jaccard_diversity_per_submission_over_all_cycles),
                                         nr_samples=NR_BOOTSTRAP_SAMPLES)
            ci_dict["Average_ci_lo"].append(lower)
            ci_dict["Average_ci_hi"].append(upper)

            shot_values_list.append(np.mean(recommendations_jaccard_diversity_per_submission_over_all_cycles))
            annotations_list.append(
                ("%.4f\n" % np.mean(recommendations_jaccard_diversity_per_submission_over_all_cycles)).replace("0.",
                                                                                                               ".") + (
                        CI_FORMAT_STRING % (lower, upper)).replace("0.", ".\!").replace(", .",
                                                                                        ",\; ."))  # "\;" a thick space; "\!" a negative thin space

        # model group average row:
        # Bootstrapped CI:
        lower, upper = bootstrapCIV6(list(all_recommendations_jaccard_diversity_per_submission_over_all_cycles),
                                     nr_samples=NR_BOOTSTRAP_SAMPLES)
        ci_dict["Average_ci_lo"].append(lower)
        ci_dict["Average_ci_hi"].append(upper)
        all_shot_values_list_for_overall_row[
            "Average"] += all_recommendations_jaccard_diversity_per_submission_over_all_cycles
        shot_values_list.append(np.mean(all_recommendations_jaccard_diversity_per_submission_over_all_cycles))
        annotations_list.append(
            ("%.4f\n" % np.mean(all_recommendations_jaccard_diversity_per_submission_over_all_cycles)).replace("0.",
                                                                                                               ".") + (
                    CI_FORMAT_STRING % (lower, upper)).replace("0.", ".\!").replace(", .",
                                                                                    ",\; ."))  # "\;" a thick space; "\!" a negative thin space

        data["Average"] = shot_values_list
        all_annotations.append(annotations_list)

        # correct model labels:
        for index in range(len(data["Model"])):
            data["Model"][index] = correctModelLabel(data["Model"][index])
        if i == 0:
            ylabel = "Tiny"
        elif i == 1:
            ylabel = "Small"
        elif i == 2:
            ylabel = "Medium"
        elif i == 3:
            ylabel = "Large"
        data[ylabel] = data["Model"]
        del data["Model"]
        data[ylabel].append("Average")
        df = pd.DataFrame(data)
        df.set_index(ylabel, inplace=True)
        df_list.append(df)
        all_annotations = np.array(all_annotations)
        all_annotations = all_annotations.T
        if values_in_percentages:
            ax = sns.heatmap(df, annot=True, fmt='.2%', cmap=cm, ax=axes[i], cbar=False,
                             annot_kws={"va": 'center_baseline'}, vmin=0.45, vmax=1)
        else:
            ax = sns.heatmap(df, annot=all_annotations, fmt="", cmap=cm, ax=axes[i], cbar=False,
                             annot_kws={"va": 'center_baseline'}, vmin=0.45, vmax=1)
        if i == 0:
            axes[i].tick_params(left=True, bottom=False, top=True, right=False, labelleft=True, labelbottom=False,
                                labeltop=True)
            axes[i].xaxis.tick_top()
            axes[i].xaxis.set_label_position('top')
        else:
            axes[i].tick_params(left=True, bottom=False, top=False, right=False, labelleft=True, labelbottom=False)
    if values_in_percentages:
        fig.colorbar(axes[i].get_children()[0], cax=axes[i + 1], orientation="horizontal",
                     format=mtick.PercentFormatter(1.0))
    else:
        fig.colorbar(axes[i].get_children()[0], cax=axes[i + 1], orientation="horizontal",
                     format=mtick.FuncFormatter(formatterWithoutLeadingZeros))

    plt.tight_layout()
    plt.savefig("result_plots/results_diversity_over_cycles_jaccard_grouped_by_shots.pdf")

    df_list = pd.concat(df_list)
    # add ci columns to csv:
    offset = 1
    for index, (col, values) in enumerate(ci_dict.items()):
        df_list.insert(index + offset, col, values, True)
        if index % 2 != 0: offset += 1
    df_list.to_csv("result_plots/results_diversity_over_cycles_jaccard_grouped_by_shots.csv")
    return


def plotDiversityOverCyclesJaccardGroupedByModelsAndShots(model_group_list, shot_group_list,
                                                          values_in_percentages=False):
    df_extraction = pd.read_csv(os.path.join(DIRECTORY, EXTRACTION_RESULTS_FILENAME))

    plt.close()
    fig, axes = plt.subplots(
        nrows=2,
        figsize=(5, 2.2),
        gridspec_kw={"height_ratios": [2.2, 0.1]})
    cm = sns.light_palette((20, 60, 50), input="husl", as_cmap=True)

    ci_dict = defaultdict(list)
    data = {
        "Model Group": ["Tiny", "Small", "Medium", "Large"]
    }
    all_annotations = list()
    for shot_label, shot_list in shot_group_list:
        df_extraction_shot = df_extraction.loc[df_extraction["shot"].isin(shot_list)].reset_index(drop=True)
        shot_values_list = list()
        annotations_list = list()
        for i, model_group in enumerate(model_group_list):
            all_recommendations_jaccard_diversity_per_submission_over_all_cycles = list()
            for model in model_group:
                df_extraction_shot_model = df_extraction_shot.loc[df_extraction_shot["model"] == model].reset_index(
                    drop=True)
                # calc jaccard only between same shots in shot_list:
                for shot in shot_list:
                    df_extraction_shot_model_shot = df_extraction_shot_model.loc[
                        df_extraction_shot_model["shot"] == shot].reset_index(drop=True)
                    for submission_id in sorted(df_extraction_shot_model_shot["submission_id"].unique()):
                        df_extraction_shot_model_shot_submission_id = df_extraction_shot_model_shot.loc[
                            df_extraction_shot_model_shot["submission_id"] == submission_id].reset_index(drop=True)
                        cycle_dict = dict()
                        for index, row in df_extraction_shot_model_shot_submission_id.iterrows():
                            movie_titles_list_without_duplicates_upper_limit = eval(
                                row["rec_list_unique_upper_limit"].lower())
                            assert (type(movie_titles_list_without_duplicates_upper_limit) is list)
                            assert (len(movie_titles_list_without_duplicates_upper_limit) == row[
                                "nr_recs_unique_upper_limit"])
                            cycle_dict[row["cycle"]] = movie_titles_list_without_duplicates_upper_limit
                        cycle_pool = set(cycle_dict)
                        cycle_pool_copy = cycle_pool.copy()
                        counter = 0
                        jaccard_list = list()
                        for cycle1 in cycle_pool_copy:
                            cycle_pool.remove(cycle1)
                            for cycle2 in cycle_pool:
                                counter += 1
                                if len(cycle_dict[cycle1]) == 0 or len(cycle_dict[cycle2]) == 0: continue
                                jaccard = len(set(cycle_dict[cycle1]) & set(cycle_dict[cycle2])) / len(
                                    set(cycle_dict[cycle1]) | set(cycle_dict[cycle2]))
                                jaccard_list.append(1 - jaccard)
                        if len(jaccard_list) == 0: continue
                        all_recommendations_jaccard_diversity_per_submission_over_all_cycles.append(
                            np.mean(jaccard_list))

            # model group average row:
            # Bootstrapped CI:
            lower, upper = bootstrapCIV6(list(all_recommendations_jaccard_diversity_per_submission_over_all_cycles),
                                         nr_samples=NR_BOOTSTRAP_SAMPLES)
            ci_dict[shot_label + "_ci_lo"].append(lower)
            ci_dict[shot_label + "_ci_hi"].append(upper)
            shot_values_list.append(np.mean(all_recommendations_jaccard_diversity_per_submission_over_all_cycles))
            annotations_list.append(
                ("%.4f\n" % np.mean(all_recommendations_jaccard_diversity_per_submission_over_all_cycles)).replace("0.",
                                                                                                                   ".") + (
                        CI_FORMAT_STRING % (lower, upper)).replace("0.", ".\!").replace(", .",
                                                                                        ",\; ."))  # "\;" a thick space; "\!" a negative thin space

        data[shot_label] = shot_values_list
        all_annotations.append(annotations_list)

    # Average column:
    shot_values_list = list()
    annotations_list = list()
    for i, model_group in enumerate(model_group_list):
        all_recommendations_jaccard_diversity_per_submission_over_all_cycles = list()
        for model in model_group:
            df_extraction_model = df_extraction.loc[df_extraction["model"] == model].reset_index(drop=True)
            for submission_id in sorted(df_extraction_model["submission_id"].unique()):
                df_extraction_model_submission_id = df_extraction_model.loc[
                    df_extraction_model["submission_id"] == submission_id].reset_index(drop=True)
                cycle_dict = dict()
                for index, row in df_extraction_model_submission_id.iterrows():
                    movie_titles_list_without_duplicates_upper_limit = eval(row["rec_list_unique_upper_limit"].lower())
                    assert (type(movie_titles_list_without_duplicates_upper_limit) is list)
                    assert (len(movie_titles_list_without_duplicates_upper_limit) == row["nr_recs_unique_upper_limit"])
                    cycle_dict[row["shot"] + str(row["cycle"])] = movie_titles_list_without_duplicates_upper_limit
                cycle_pool = set(cycle_dict)
                cycle_pool_copy = cycle_pool.copy()
                counter = 0
                jaccard_list = list()
                for cycle1 in cycle_pool_copy:
                    cycle_pool.remove(cycle1)
                    for cycle2 in cycle_pool:
                        counter += 1
                        if len(cycle_dict[cycle1]) == 0 or len(cycle_dict[cycle2]) == 0: continue
                        jaccard = len(set(cycle_dict[cycle1]) & set(cycle_dict[cycle2])) / len(
                            set(cycle_dict[cycle1]) | set(cycle_dict[cycle2]))
                        intersection = len(set(cycle_dict[cycle1]).intersection(set(cycle_dict[cycle2])))
                        union = len(set(cycle_dict[cycle1]).union(set(cycle_dict[cycle2])))
                        assert (intersection / union) == jaccard
                        jaccard_list.append(1 - jaccard)
                if len(jaccard_list) == 0: continue
                all_recommendations_jaccard_diversity_per_submission_over_all_cycles.append(np.mean(jaccard_list))

        # model group average row:
        # Bootstrapped CI:
        lower, upper = bootstrapCIV6(list(all_recommendations_jaccard_diversity_per_submission_over_all_cycles),
                                     nr_samples=NR_BOOTSTRAP_SAMPLES)
        ci_dict["Average_ci_lo"].append(lower)
        ci_dict["Average_ci_hi"].append(upper)
        shot_values_list.append(np.mean(all_recommendations_jaccard_diversity_per_submission_over_all_cycles))
        annotations_list.append(
            ("%.4f\n" % np.mean(all_recommendations_jaccard_diversity_per_submission_over_all_cycles)).replace("0.",
                                                                                                               ".") + (
                    CI_FORMAT_STRING % (lower, upper)).replace("0.", ".\!").replace(", .",
                                                                                    ",\; ."))  # "\;" a thick space; "\!" a negative thin space

    data["Average"] = shot_values_list
    all_annotations.append(annotations_list)
    df = pd.DataFrame(data)
    df.set_index("Model Group", inplace=True)
    all_annotations = np.array(all_annotations)
    all_annotations = all_annotations.T
    if values_in_percentages:
        sns.heatmap(df, annot=True, fmt='.2%', cmap=cm, ax=axes[0], cbar=False, annot_kws={"va": 'center_baseline'},
                    vmin=0.59, vmax=0.91)
    else:
        sns.heatmap(df, annot=all_annotations, fmt="", cmap=cm, ax=axes[0], cbar=False,
                    annot_kws={"va": 'center_baseline'}, vmin=0.59, vmax=0.91)
    axes[0].set_ylabel("", rotation=0, labelpad=100)
    axes[0].tick_params(left=True, bottom=False, top=True, right=False, labelleft=True, labelbottom=False,
                        labeltop=True)
    axes[0].xaxis.tick_top()
    axes[0].xaxis.set_label_position('top')
    if values_in_percentages:
        fig.colorbar(axes[0].get_children()[0], cax=axes[1], orientation="horizontal",
                     format=mtick.PercentFormatter(1.0))
    else:
        fig.colorbar(axes[0].get_children()[0], cax=axes[1], orientation="horizontal",
                     format=mtick.FuncFormatter(formatterWithoutLeadingZeros))

    plt.tight_layout()
    plt.savefig("result_plots/results_diversity_over_cycles_jaccard_grouped_by_models_and_shots.pdf")

    # add ci columns to csv:
    offset = 1
    for index, (col, values) in enumerate(ci_dict.items()):
        df.insert(index + offset, col, values, True)
        if index % 2 != 0: offset += 1
    df.to_csv("result_plots/results_diversity_over_cycles_jaccard_grouped_by_models_and_shots.csv")
    return


def plotYears(df_extraction, model_list, shot_list, values_in_percentages=False):
    plt.close()
    fig, axes = plt.subplots(
        len(df_extraction["model"].unique()),
        len(df_extraction["shot"].unique()) + 1,
        figsize=(10, 15), sharex=True, sharey=False)

    all_years_overall = list()
    for i, model in enumerate(model_list):
        df_extraction_model = df_extraction.loc[df_extraction["model"] == model].reset_index(drop=True)
        all_years = list()
        for j, (shot, shot_label) in enumerate(shot_list):
            df_extraction_model_shot = df_extraction_model.loc[df_extraction_model["shot"] == shot].reset_index(
                drop=True)
            years = list()
            for index, row in df_extraction_model_shot.iterrows():
                movie_titles_list = eval(row["rec_list"].lower())
                for movie_title in movie_titles_list:
                    year = movie_title.split("(")[-1].split(")")[0].strip()
                    years.append(int(year))
                    all_years.append(int(year))
                    all_years_overall.append(int(year))
            if i == 0:
                axes[i, j].text(x=0.5, y=1.1, s=shot_label, ha='center', va='bottom', transform=axes[i, j].transAxes)
            if j == 0:
                axes[i, j].text(x=-0.5, y=0.3, s=correctModelLabel(model), ha='right', va='bottom',
                                transform=axes[i, j].transAxes)
            else:
                axes[i, j].tick_params(labelleft=False)
            if len(years) == 0: continue
            sns.histplot(years, binwidth=5, ax=axes[i, j], stat='probability')
            axes[i, j].set_xlim(1900, 2025)
            if model.startswith("qwen2:0.5b"):
                axes[i, j].set_ylim(0, 0.62)
            else:
                axes[i, j].set_ylim(0, 0.32)
            axes[i, j].set_ylabel("", rotation=0, labelpad=100)
            axes[i, j].set_xticks([1900, 1925, 1950, 1975, 2000, 2025])
            plt.setp(axes[i, j].xaxis.get_majorticklabels(), rotation=90)
            axes[i, j].set_xlabel('Year')
            if values_in_percentages:
                axes[i, j].yaxis.set_major_formatter(mtick.PercentFormatter(1.0, 0))

        # Average column:
        if i == 0:
            axes[i, j + 1].text(x=0.5, y=1.1, s="Average", ha='center', va='bottom', transform=axes[i, j + 1].transAxes,
                                style="italic")  # weight='bold'
        sns.histplot(all_years, binwidth=5, ax=axes[i, j + 1], stat='probability')
        axes[i, j + 1].set_xlim(1900, 2025)
        if model.startswith("qwen2:0.5b"):
            axes[i, j + 1].set_ylim(0, 0.62)
        else:
            axes[i, j + 1].set_ylim(0, 0.32)
        axes[i, j + 1].set_ylabel("", rotation=0, labelpad=100)
        axes[i, j + 1].set_xticks([1900, 1925, 1950, 1975, 2000, 2025])
        plt.setp(axes[i, j + 1].xaxis.get_majorticklabels(), rotation=90)
        axes[i, j + 1].set_xlabel('Year')
        axes[i, j + 1].tick_params(labelleft=False)

    plt.tight_layout()
    plt.savefig("result_plots/results_years_detailed.pdf")

    f = open("result_plots/results_years.csv", "w")
    f.write("year,count\n")
    for year in sorted(set(all_years_overall)):
        f.write("%s,%d\n" % (str(year), all_years_overall.count(year)))
    f.close()
    return


def plotYearsGrouped(df_extraction, model_group_list, shot_list, values_in_percentages=False):
    height_ratios = list()
    for model_group in model_group_list:
        height_ratios += [1] * len(model_group) + [0.5]
    height_ratios = height_ratios[:-1]
    rows_to_delete = list()

    plt.close()
    fig, axes = plt.subplots(
        nrows=len(height_ratios),
        ncols=len(df_extraction["shot"].unique()),
        figsize=(13.5, 15), sharex=False, sharey=False,
        gridspec_kw={'height_ratios': height_ratios})

    all_years_overall = list()
    i = -1
    for mg, model_group in enumerate(model_group_list):
        for m, model in enumerate(model_group):
            i += 1
            df_extraction_model = df_extraction.loc[df_extraction["model"] == model].reset_index(drop=True)
            all_years = list()
            for j, (shot, shot_label) in enumerate(shot_list):
                df_extraction_model_shot = df_extraction_model.loc[df_extraction_model["shot"] == shot].reset_index(
                    drop=True)
                years = list()
                for index, row in df_extraction_model_shot.iterrows():
                    movie_titles_list = eval(row["rec_list"].lower())
                    for movie_title in movie_titles_list:
                        year = movie_title.split("(")[-1].split(")")[0].strip()
                        years.append(int(year))
                        all_years.append(int(year))
                        all_years_overall.append(int(year))
                if i == 0:
                    axes[i, j].text(x=0.5, y=1.1, s=shot_label, ha='center', va='bottom',
                                    transform=axes[i, j].transAxes)
                if j == 0:
                    axes[i, j].text(x=-0.4, y=0.3, s=correctModelLabel(model), ha='right', va='bottom',
                                    transform=axes[i, j].transAxes)
                else:
                    axes[i, j].tick_params(labelleft=False)
                if len(years) == 0: continue
                sns.histplot(years, binwidth=5, ax=axes[i, j], stat='probability')
                axes[i, j].set_xlim(1900, 2025)
                if model.startswith("qwen2:0.5b") or model.startswith("qwen2.5:0.5b"):
                    axes[i, j].set_ylim(0, 0.62)
                else:
                    axes[i, j].set_ylim(0, 0.32)
                axes[i, j].set_ylabel("", rotation=0, labelpad=100)
                axes[i, j].set_xticks([])
                plt.setp(axes[i, j].patches, linewidth=0.5)
                if values_in_percentages:
                    axes[i, j].yaxis.set_major_formatter(mtick.PercentFormatter(1.0, 0))
        i += 1
        rows_to_delete.append(i)
    rows_to_delete = rows_to_delete[:-1]
    for row in rows_to_delete:
        for ax in axes[row, :]:
            fig.delaxes(ax)

    for j, (shot, shot_label) in enumerate(shot_list):
        axes[-1, j].set_xticks([1900, 1925, 1950, 1975, 2000, 2025])
        axes[-1, j].set_xlabel('Year')
        plt.setp(axes[-1, j].xaxis.get_majorticklabels(), rotation=90)

    plt.tight_layout()
    plt.subplots_adjust(left=0.17, wspace=0.2, hspace=0.2)
    axes[5, 0].text(x=-1.1, y=0.5, s="Tiny", ha='center', va='center', transform=axes[5, 0].transAxes, rotation=90)
    axes[17, 0].text(x=-1.1, y=1.2, s="Small", ha='center', va='center', transform=axes[17, 0].transAxes, rotation=90)
    axes[27, 0].text(x=-1.3, y=0.5, s="Medium", ha='center', va='center', transform=axes[27, 0].transAxes, rotation=90)
    axes[37, 0].text(x=-1.4, y=1.2, s="Large", ha='center', va='center', transform=axes[37, 0].transAxes, rotation=90)
    axes[20, 0].text(x=-1.5, y=0.5, s="Release Years of Movies", ha='center', va='center',
                     transform=axes[20, 0].transAxes, rotation=90, fontsize=14)
    plt.savefig("result_plots/results_years_detailed_grouped.pdf")
    return


def checkFewShotExamplesOverlapWithRecommendations(model_list, shot_list):
    df_extraction = pd.read_csv(os.path.join(DIRECTORY, EXTRACTION_RESULTS_FILENAME))
    df_examples_overlap = pd.DataFrame(index=model_list, columns=[shot for shot, shot_label in shot_list])
    for col in df_examples_overlap.columns:
        df_examples_overlap[col].values[:] = 0
    target_shot = "shot_few_v1_10_examples"
    for index, row in df_extraction.iterrows():
        if target_shot not in row["shot"]: continue
        movie_titles_list_without_duplicates_upper_limit = eval(row["rec_list_unique_upper_limit"].lower())
        assert (type(movie_titles_list_without_duplicates_upper_limit) is list)
        assert (len(movie_titles_list_without_duplicates_upper_limit) == row["nr_recs_unique_upper_limit"])
        found_path = None
        for results_directory_prefix in RESULTS_DIRECTORY_PREFIXES:
            for cycle_directory_prefix in CYCLE_DIRECTORY_PREFIXES:
                path = os.path.join(RESULTS_DIRECTORY,
                                    results_directory_prefix + row["shot"],
                                    cycle_directory_prefix + row["model"] + "__" + row["shot"] + (
                                            "__cycle_%02d" % row["cycle"]),
                                    row["submission_id"] + "__" + row["model"] + "_" + REQUEST_FILES_SUFFIX
                                    )
                if os.path.exists(path):
                    found_path = path
                    break
        if found_path is None:
            return
        with open(found_path) as f:
            request_content = f.read()

        for movie in movie_titles_list_without_duplicates_upper_limit:
            if movie in request_content:
                df_examples_overlap.at[row["model"], row["shot"]] += 1

                # check if movie appears in same submission in other shots:
                df_extraction_model = df_extraction.loc[(df_extraction["model"] == row["model"]) & (
                        df_extraction["submission_id"] == row["submission_id"])].reset_index(drop=True)
                for shot, shot_label in shot_list:
                    if shot == target_shot: continue
                    df_extraction_model_shot = df_extraction_model.loc[df_extraction_model["shot"] == shot].reset_index(
                        drop=True)
                    for temp_index, temp_row in df_extraction_model_shot.iterrows():
                        temp_movie_titles_list = eval(temp_row["rec_list_unique_upper_limit"].lower())
                        if movie in temp_movie_titles_list:
                            df_examples_overlap.at[row["model"], shot] += 1 / 3.0
    df_examples_overlap.to_csv(os.path.join(DIRECTORY, "result_plots", "examples_overlap__few_shot_10_examples.csv"),
                               index=True)
    return


def main():
    ####################################################################################################################
    # execute only once!
    createMovieTitlesCSVs()
    # createMovieTitlesCSVs(data_directory=os.path.join(DATA_DIRECTORY, "data_2024_08")) # only for robustness analysis

    # execute only once!
    createAndSaveSubmission2trueMovieTitles()

    # submissionid_parameters = loadPickle(os.path.join(DATA_DIRECTORY, "data_2024_08/only_fitting", SUBMISSIONID_PARAMETERS_FILENAME))
    # submissionid_commentsmatched = loadPickle(os.path.join(DATA_DIRECTORY, "data_2024_08/only_fitting", SUBMISSIONID_COMMENTSMATCHED_FILENAME))
    # createAndSaveSubmission2trueMovieTitles(data_directory=os.path.join(DATA_DIRECTORY, "data_2024_08"),
    #                                         submissionid_parameters=submissionid_parameters,
    #                                         submissionid_commentsmatched=submissionid_commentsmatched) # only for robustness analysis
    ####################################################################################################################
    extractRecommendations()
    evaluateRecommendations(multi_language=True)
    # evaluateRecommendations(data_directory=os.path.join(DATA_DIRECTORY, "data_2024_08"), multi_language=True) # only for robustness analysis
    ####################################################################################################################
    getMedianCycles("exact_match_f1_score", multi_language=True)
    ####################################################################################################################
    computeANOVA("exact_match_f1_score", multi_language=True)
    # computeANOVA("threshold_match_f1_score", multi_language=True)
    ####################################################################################################################
    df_extraction = pd.read_csv(os.path.join(DIRECTORY, EXTRACTION_RESULTS_MEDIAN_CYCLES_FILENAME))
    model_list = sorted(df_extraction["model"].unique())
    model_group_list = [  # sorted by size (grouped) and then alphabetically
        [  # tiny:
            'gemma:2b-instruct-v1.1-q4_0',
            'gemma2:2b-instruct-q4_0',
            "llama3.2:1b-instruct-q4_0",
            "llama3.2:3b-instruct-q4_0",
            'phi3:3.8b-mini-128k-instruct-q4_0',
            'phi3.5:3.8b-mini-instruct-q4_0',
            'qwen2:0.5b-instruct-q4_0',
            'qwen2:1.5b-instruct-q4_0',
            "qwen2.5:0.5b-instruct-q4_0",
            "qwen2.5:1.5b-instruct-q4_0",
            "qwen2.5:3b-instruct-q4_0",
        ],
        [  # small:
            'gemma:7b-instruct-v1.1-q4_0',
            'gemma2:9b-instruct-q4_0',
            'gpt-4o-mini-2024-07-18',
            'llama3:8b-instruct-q4_0',
            'llama3.1:8b-instruct-q4_0',
            'mistral:7b-instruct-v0.3-q4_0',
            'qwen2:7b-instruct-q4_0',
            "qwen2.5:7b-instruct-q4_0",
            'yi:6b-chat-v1.5-q4_0',
            'yi:9b-chat-v1.5-q4_0',
        ],
        [  # medium:
            'gemma2:27b-instruct-q4_0',
            'gpt-3.5-turbo-0125',
            'mistral-nemo:12b-instruct-2407-q4_0',
            "mistral-small:22b-instruct-2409-q4_0",
            'mixtral:8x7b-instruct-v0.1-q4_0',
            'phi3:14b-medium-128k-instruct-q4_0',
            "qwen2.5:14b-instruct-q4_0",
            "qwen2.5:32b-instruct-q4_0",
            'yi:34b-chat-v1.5-q4_0',
        ],
        [  # large:
            'gpt-4o-2024-08-06',
            'llama3:70b-instruct-q4_0',
            'llama3.1:70b-instruct-q4_0',
            'llama3.1:405b-instruct-q4_0',
            'mistral-large:123b-instruct-2407-q4_0',
            'mixtral:8x22b-instruct-v0.1-q4_0',
            'qwen2:72b-instruct-q4_0',
            "qwen2.5:72b-instruct-q4_0",
        ],
    ]

    shot_list = [
        # (shot name, shot label for plot)
        ('shot_zero', "Zero-Shot"),
        ('shot_identity_reddit', "Identity:\nReddit User"),
        ('shot_identity_movie_critic', "Identity:\nMovie Critic"),
        ('shot_identity_recommender', "Identity:\nMovie Rec."),
        ('shot_few_v1_1_examples', "Few-Shot:\n1 Example"),
        ('shot_few_v1_5_examples', "Few-Shot:\n5 Examples"),
        ('shot_few_v1_10_examples', "Few-Shot:\n10 Examples"),
    ]

    shot_group_list = [
        # (shot label for plot, [shot name 1, shot name 2,...] )
        ("Zero-Shot", ['shot_zero']),
        ("Identity", ['shot_identity_reddit', 'shot_identity_movie_critic', 'shot_identity_recommender']),
        ("Few-Shot", ['shot_few_v1_1_examples', 'shot_few_v1_5_examples', 'shot_few_v1_10_examples']),
    ]

    ####################################################################################################################
    # plot recommendation accuracy statistics:

    createAccuracyHeatmapGrouped(model_group_list, shot_list, "exact_match_f1_score", multi_language=True,
                                 values_in_percentages=False)

    createAccuracyHeatmapGroupedByShots(model_group_list, shot_group_list, "exact_match_f1_score", multi_language=True,
                                        values_in_percentages=False)
    createAccuracyHeatmapGroupedByShots(model_group_list, shot_group_list, "exact_match_precision", multi_language=True,
                                        values_in_percentages=False)
    createAccuracyHeatmapGroupedByShots(model_group_list, shot_group_list, "exact_match_recall", multi_language=True,
                                        values_in_percentages=False)
    createAccuracyHeatmapGroupedByShots(model_group_list, shot_group_list, "exact_match_nDCG", multi_language=True,
                                        values_in_percentages=False)

    createAccuracyHeatmapGroupedByModelsAndShots(model_group_list, shot_group_list, "exact_match_f1_score",
                                                 multi_language=True, values_in_percentages=False)
    createAccuracyHeatmapGroupedByModelsAndShots(model_group_list, shot_group_list, "exact_match_precision",
                                                 multi_language=True, values_in_percentages=False)
    createAccuracyHeatmapGroupedByModelsAndShots(model_group_list, shot_group_list, "exact_match_recall",
                                                 multi_language=True, values_in_percentages=False)
    createAccuracyHeatmapGroupedByModelsAndShots(model_group_list, shot_group_list, "exact_match_nDCG",
                                                 multi_language=True, values_in_percentages=False)

    ####################################################################################################################
    # plot response statistics:

    plotValidJsonGrouped(df_extraction, model_group_list, shot_list, values_in_percentages=False)
    plotValidJsonGroupedByShots(df_extraction, model_group_list, shot_group_list, values_in_percentages=False)
    plotValidJsonGroupedByModelsAndShots(df_extraction, model_group_list, shot_group_list, values_in_percentages=False)

    # calc the nr of recs only on valid json responses:
    df_extraction_only_valid_jsons = df_extraction.loc[df_extraction['valid_json'] == True].reset_index(drop=True)
    plotNrRecsHeatmapGrouped(df_extraction_only_valid_jsons, model_group_list, shot_list, values_in_percentages=False)
    plotNrRecsOnly10HeatmapGrouped(df_extraction_only_valid_jsons, model_group_list, shot_list,
                                   values_in_percentages=False)
    plotNrRecsOnly10HeatmapGroupedByShots(df_extraction_only_valid_jsons, model_group_list, shot_group_list,
                                          values_in_percentages=False)
    plotNrRecsOnly10HeatmapGroupedByModelsAndShots(df_extraction_only_valid_jsons, model_group_list, shot_group_list,
                                                   values_in_percentages=False)

    plotUniqueMoviesGrouped(df_extraction, model_group_list, shot_list, values_in_percentages=False)
    plotUniqueMoviesGroupedByShots(df_extraction, model_group_list, shot_group_list, values_in_percentages=False)
    plotUniqueMoviesGroupedByModelsAndShots(df_extraction, model_group_list, shot_group_list,
                                            values_in_percentages=False)

    plotCorrectYearsGrouped(df_extraction, model_group_list, shot_list, values_in_percentages=False)
    plotCorrectYearsGroupedByShots(df_extraction, model_group_list, shot_group_list, values_in_percentages=False)
    plotCorrectYearsGroupedByModelsAndShots(df_extraction, model_group_list, shot_group_list,
                                            values_in_percentages=False)

    plotDiversityOverCyclesJaccardGrouped(model_group_list, shot_list, values_in_percentages=False)
    plotDiversityOverCyclesJaccardGroupedByShots(model_group_list, shot_group_list, values_in_percentages=False)
    plotDiversityOverCyclesJaccardGroupedByModelsAndShots(model_group_list, shot_group_list,
                                                          values_in_percentages=False)

    plotYears(df_extraction, model_list, shot_list, values_in_percentages=False)
    plotYearsGrouped(df_extraction, model_group_list, shot_list, values_in_percentages=False)

    checkFewShotExamplesOverlapWithRecommendations(model_list, shot_list)
    ####################################################################################################################
    return


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    debug_msg("Started %s" % str(sys.argv[0]), DebugLevel.DEBUG)

    main()

    end_time = datetime.datetime.now()
    debug_msg("Finished %s\tDuration: %s" % (str(sys.argv[0]), getDurationString(start_time, end_time)),
              DebugLevel.DEBUG)
